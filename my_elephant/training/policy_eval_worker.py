"""
策略评估 HTTP 子进程入口：``python -m my_elephant.training.policy_eval_worker --checkpoint ... --port 17890``

绑定 ``127.0.0.1``，提供 ``GET /health``、``POST /eval``（JSON ``{"fen": "..."}``）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from my_elephant.training.play_model_loader import load_successor_policy_for_play
from my_elephant.training.policy_eval_http import run_eval_http_server


def main() -> None:
    p = argparse.ArgumentParser(description="策略网络本地 HTTP 评估子进程")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--in-channels", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cpu")
    model, flist = load_successor_policy_for_play(
        args.checkpoint, device, in_channels=args.in_channels
    )
    run_eval_http_server(args.host, int(args.port), model, device, flist)


if __name__ == "__main__":
    main()
