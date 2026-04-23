"""棋谱样本：`IterableDataset` + `DataLoader` 多进程预取 (后继平面, mask, 着法下标)。"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from my_elephant.chess import FEATURE_LIST, convert_game
from my_elephant.chess.rationale import POLICY_MAX_LEGAL_MOVES

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


def _pad_one_sample(
    planes_kchw: np.ndarray,
    k_valid: int,
    label_idx: int,
    k_max: int,
    red_to_move: bool,
    current_chw: np.ndarray,
    outcome_cls: int,
) -> tuple[np.ndarray, np.ndarray, np.int64, bool, np.ndarray, np.int64]:
    if k_valid > k_max:
        raise ValueError(
            f"合法着法数 {k_valid} 超过 POLICY_MAX_LEGAL_MOVES={k_max}，请在 rationale 中调大 POLICY_MAX_LEGAL_MOVES"
        )
    c = int(planes_kchw.shape[1])
    pad = np.zeros((k_max, c, 10, 9), dtype=np.float32)
    pad[:k_valid] = planes_kchw
    mask = np.zeros((k_max,), dtype=np.bool_)
    mask[:k_valid] = True
    x_hwc = np.transpose(pad, (0, 2, 3, 1))
    cur_hwc = np.transpose(current_chw, (1, 2, 0))
    return x_hwc, mask, np.int64(label_idx), red_to_move, cur_hwc, np.int64(outcome_cls)


def collate_successor_policy_value_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.int64, bool, np.ndarray, np.int64]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.stack([b[0] for b in batch], axis=0)
    ms = np.stack([b[1] for b in batch], axis=0)
    ys = np.stack([np.int64(b[2]) for b in batch], axis=0)
    red = np.asarray([bool(b[3]) for b in batch], dtype=np.bool_)
    cur = np.stack([b[4] for b in batch], axis=0)
    yv = np.stack([np.int64(b[5]) for b in batch], axis=0)
    return xs, ms, ys, red, cur, yv


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
    # 文件数少于 worker 数时，按 id 间隔取，避免空 shard
    sub = [paths[i] for i in range(wi.id, n, wi.num_workers)]
    return sub if sub else paths


class SuccessorPolicyIterableDataset(IterableDataset):
    """
    无限遍历：各 worker 只处理自己分片内的棋谱路径，打乱后循环；
    每条棋谱用 `convert_game` 逐步 yield，经填充后作为单样本供 DataLoader 组 batch。

    ``sources`` 可为单列无表头 CSV 路径，或已解析好的棋谱路径列表。
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
        self.k_max = POLICY_MAX_LEGAL_MOVES

    def __iter__(self) -> Any:
        my_files = _shard_filelist_for_worker(self.filelist)
        while True:
            rnd = random.Random()
            rnd.shuffle(my_files)
            for path in my_files:
                try:
                    for (
                        planes_kchw,
                        k_valid,
                        label_idx,
                        red_to_move,
                        current_chw,
                        outcome_cls,
                    ) in convert_game(path, feature_list=self.feature_list):
                        yield _pad_one_sample(
                            planes_kchw,
                            k_valid,
                            label_idx,
                            self.k_max,
                            red_to_move,
                            current_chw,
                            outcome_cls,
                        )
                except Exception:
                    continue


def make_policy_dataloader(
    sources: PolicySources,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    ``num_workers>0`` 时在子进程内解析棋谱、枚举着法，主进程组 batch 后送 GPU。
    ``pin_memory=True``（CUDA 训练时建议）可与 ``non_blocking=True`` 拷贝配合。
    ``prefetch_factor`` 为每个 worker 预取的 batch 数（仅 ``num_workers>0`` 时生效）。
    """
    ds = SuccessorPolicyIterableDataset(sources)
    kw: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_successor_policy_value_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(**kw)
