"""棋谱样本：`IterableDataset` + `DataLoader` 多进程并行解析与预取（两阶段：起点格 + 落点格 + 价值标签）。"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from my_elephant.chess import FEATURE_LIST, convert_game

PolicySources = str | Path | list[str]


def discover_cbf_files(root: Path | str, *, recursive: bool = True) -> list[str]:
    """在 ``root`` 下搜集所有 ``.cbf`` 路径（排序后返回，元素为 resolve 后的字符串）。"""
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise NotADirectoryError(str(r))
    paths: list[Path] = []
    if recursive:
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".cbf":
                paths.append(p)
    else:
        for p in r.iterdir():
            if p.is_file() and p.suffix.lower() == ".cbf":
                paths.append(p)
    out = sorted(str(p.resolve()) for p in paths)
    if not out:
        scope = "（含子目录）" if recursive else "（仅一层）"
        raise FileNotFoundError(f"在 {r}{scope} 未找到 .cbf 文件")
    return out


def split_paths_train_test(
    paths: list[str], train_ratio: float, *, seed: int = 42
) -> tuple[list[str], list[str]]:
    """与 ``data_prep.split_manifest`` 一致的随机划分：shuffle 后按 ``int(n * train_ratio)`` 切开。"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 应在 (0,1) 内，收到 {train_ratio}")
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    gap = int(len(shuffled) * train_ratio)
    if gap <= 0 or gap >= len(shuffled):
        raise ValueError(
            f"划分后 train 或 test 为空（n={len(shuffled)}, train_ratio={train_ratio}）"
        )
    return shuffled[:gap], shuffled[gap:]


def _policy_dataloader_worker_init(_worker_id: int) -> None:
    """子进程内限制 BLAS/OpenMP 线程数，避免多 worker 与主训练线程抢核导致变慢。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def collate_twohead_policy_value_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.int64, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur = np.stack([b[0] for b in batch], axis=0)
    msrc = np.stack([b[1] for b in batch], axis=0)
    mdst = np.stack([b[2] for b in batch], axis=0)
    ys = np.stack([np.int64(b[3]) for b in batch], axis=0)
    yd = np.stack([np.int64(b[4]) for b in batch], axis=0)
    yv = np.stack([np.int64(b[5]) for b in batch], axis=0)
    return cur, msrc, mdst, ys, yd, yv


def _shard_filelist_for_worker(filelist: list[str]) -> list[str]:
    wi = get_worker_info()
    paths = list(filelist)
    if not paths:
        return paths
    if wi is None:
        return paths
    n = len(paths)
    per = int(math.ceil(n / float(wi.num_workers)))
    start = wi.id * per
    end = min(start + per, n)
    if start < end:
        return paths[start:end]
    sub = [paths[i] for i in range(wi.id, n, wi.num_workers)]
    return sub if sub else paths


class SuccessorPolicyIterableDataset(IterableDataset):
    """
    无限遍历：各 worker 分片棋谱路径；``convert_game`` 每步 yield
    当前局面 HWC、起点/落点 mask 与标签，供两阶段策略 + 价值头训练。
    """

    def __init__(self, sources: PolicySources) -> None:
        super().__init__()
        if isinstance(sources, list):
            self.filelist = [str(x) for x in sources]
        else:
            df = pd.read_csv(sources, header=None, index_col=None)
            self.filelist = df.iloc[:, 0].astype(str).tolist()
        if not self.filelist:
            label = sources if isinstance(sources, (str, Path)) else "filelist"
            raise ValueError(f"棋谱清单为空: {label}")
        self.feature_list: dict[str, list[str]] = {
            "red": list(FEATURE_LIST["red"]),
            "black": list(FEATURE_LIST["black"]),
        }

    def __iter__(self) -> Any:
        my_files = _shard_filelist_for_worker(self.filelist)
        while True:
            rnd = random.Random()
            rnd.shuffle(my_files)
            for path in my_files:
                try:
                    for sample in convert_game(path, feature_list=self.feature_list):
                        yield sample
                except Exception:
                    continue


def make_policy_dataloader(
    sources: PolicySources,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    *,
    drop_last: bool = False,
    pin_memory_device: str | None = None,
) -> DataLoader:
    ds = SuccessorPolicyIterableDataset(sources)
    kw: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_twohead_policy_value_batch,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
        kw["worker_init_fn"] = _policy_dataloader_worker_init
        if sys.platform == "win32":
            kw["multiprocessing_context"] = "spawn"
    if pin_memory and pin_memory_device is not None:
        try:
            kw["pin_memory_device"] = pin_memory_device
        except TypeError:
            pass
    return DataLoader(**kw)


def build_policy_train_val_loaders(
    train_sources: PolicySources,
    val_sources: PolicySources,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    pin_memory_device: str | None = None,
    *,
    train_drop_last: bool = True,
    val_drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = make_policy_dataloader(
        train_sources,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=train_drop_last,
        pin_memory_device=pin_memory_device,
    )
    val_loader = make_policy_dataloader(
        val_sources,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=val_drop_last,
        pin_memory_device=pin_memory_device,
    )
    return train_loader, val_loader


def default_num_workers(cap: int = 8) -> int:
    try:
        n = os.cpu_count() or 2
    except NotImplementedError:
        n = 2
    return max(0, min(cap, n))
