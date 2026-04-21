"""
加载 PyTorch 策略 checkpoint，命令行对弈（合法着法后继局面打分，argmax）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from my_elephant.chess import FEATURE_LIST, GamePlay
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS
from my_elephant.chess.xml_samples import successor_planes_for_legals
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    batched_successors_nhwc_to_torch,
    logits_as_red_preference,
    torch_load_checkpoint,
)


def neural_move(gameplay: GamePlay, model: SuccessorPolicy, device: torch.device) -> None:
    legals = sorted(gameplay.legal_moves_iccs())
    if not legals:
        raise RuntimeError("当前局面无合法着法")

    flist: dict[str, list[str]] = {
        "red": list(FEATURE_LIST["red"]),
        "black": list(FEATURE_LIST["black"]),
    }
    planes_kchw = successor_planes_for_legals(gameplay.bb, legals, flist)
    x_hwc = np.transpose(planes_kchw, (0, 2, 3, 1))
    x = np.expand_dims(x_hwc, axis=0)
    xt = batched_successors_nhwc_to_torch(x, device)
    logits = model.predict_move_logits(xt)
    red = gameplay.get_side() == "red"
    logits_r = logits_as_red_preference(logits, red)
    li = int(torch.argmax(logits_r).item())
    best_mv = legals[li]
    mv = f"{best_mv[0]}{best_mv[1]}-{best_mv[2]}{best_mv[3]}"
    print(f"神经网络 ({gameplay.get_side()}): {mv}")
    gameplay.make_move(mv)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="与 PyTorch 策略网络对弈")
    p.add_argument("--checkpoint", type=Path, required=True, help="epoch_*.pt 文件路径")
    p.add_argument("--gpu", type=int, default=0, help="-1 为 CPU")
    p.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="与 checkpoint 一致；缺省从 ckpt 读取",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    ckpt = torch_load_checkpoint(args.checkpoint, device)
    num_res = int(ckpt.get("num_res_layers", 10))
    in_ch = (
        args.in_channels
        if args.in_channels is not None
        else int(ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS)))
    )
    model = SuccessorPolicy(num_res_layers=num_res, in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gameplay = GamePlay()
    print("当前轮到:", gameplay.get_side())
    print("输入走法如 77-47；直接回车=由网络走当前这一方；q 退出。")

    while True:
        gameplay.print_board()
        line = input("> ").strip()
        if line.lower() == "q":
            break
        if not line:
            neural_move(gameplay, model, device)
        else:
            gameplay.make_move(line)


if __name__ == "__main__":
    main()
