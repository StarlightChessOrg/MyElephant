"""将目录中的 .cbf 棋谱随机划分为 train/test 清单 CSV（单列绝对或相对路径）。"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def write_train_test_lists(
    cbf_dir: Path,
    train_csv: Path,
    test_csv: Path,
    train_ratio: float = 0.9,
    seed: int | None = None,
) -> tuple[int, int]:
    if seed is not None:
        random.seed(seed)
    allfiles = sorted(cbf_dir.iterdir())
    paths = [str(p.resolve()) for p in allfiles if p.is_file() and p.suffix.lower() == ".cbf"]
    if not paths:
        raise FileNotFoundError(f"目录中未找到 .cbf 文件: {cbf_dir}")
    random.shuffle(paths)
    gap = int(len(paths) * train_ratio)
    trainfiles = paths[:gap]
    testfiles = paths[gap:]
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trainfiles).to_csv(train_csv, header=False, index=False, encoding="utf-8")
    pd.DataFrame(testfiles).to_csv(test_csv, header=False, index=False, encoding="utf-8")
    return len(trainfiles), len(testfiles)


def main() -> None:
    p = argparse.ArgumentParser(description="生成 data/train_list.csv 与 data/test_list.csv")
    p.add_argument("--cbf-dir", type=Path, default=Path("data/imsa-cbf"))
    p.add_argument("--train-out", type=Path, default=Path("data/train_list.csv"))
    p.add_argument("--test-out", type=Path, default=Path("data/test_list.csv"))
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    n_train, n_test = write_train_test_lists(
        args.cbf_dir, args.train_out, args.test_out, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"已写入 {args.train_out} ({n_train}) 与 {args.test_out} ({n_test})")


if __name__ == "__main__":
    main()
