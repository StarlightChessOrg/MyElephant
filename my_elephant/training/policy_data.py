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


def _pad_one_sample(
    planes_kchw: np.ndarray,
    k_valid: int,
    label_idx: int,
    k_max: int,
    red_to_move: bool,
) -> tuple[np.ndarray, np.ndarray, np.int64, bool]:
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
    return x_hwc, mask, np.int64(label_idx), red_to_move


def collate_successor_policy_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.int64, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.stack([b[0] for b in batch], axis=0)
    ms = np.stack([b[1] for b in batch], axis=0)
    ys = np.stack([np.int64(b[2]) for b in batch], axis=0)
    red = np.asarray([bool(b[3]) for b in batch], dtype=np.bool_)
    return xs, ms, ys, red


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
    """

    def __init__(self, datafile: str | Path) -> None:
        super().__init__()
        df = pd.read_csv(datafile, header=None, index_col=None)
        self.filelist: list[str] = df.iloc[:, 0].astype(str).tolist()
        if not self.filelist:
            raise ValueError(f"棋谱清单为空: {datafile}")
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
                    for planes_kchw, k_valid, label_idx, red_to_move in convert_game(
                        path, feature_list=self.feature_list
                    ):
                        yield _pad_one_sample(planes_kchw, k_valid, label_idx, self.k_max, red_to_move)
                except Exception:
                    continue


def make_policy_dataloader(
    csv_path: str | Path,
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
    ds = SuccessorPolicyIterableDataset(csv_path)
    kw: dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_successor_policy_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(**kw)
