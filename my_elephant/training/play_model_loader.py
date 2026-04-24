"""从 checkpoint 构建对弈用 ``SuccessorPolicy``（供 ``play_policy_torch`` 与 HTTP 评估子进程复用）。"""

from __future__ import annotations

from pathlib import Path

import torch

from my_elephant.chess import FEATURE_LIST
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    count_resnet_blocks_in_state,
    torch_load_checkpoint,
)


def _infer_filters_from_state(sd: dict) -> int:
    w = sd.get("stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    return 256


def load_successor_policy_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[SuccessorPolicy, dict[str, list[str]]]:
    ckpt = torch_load_checkpoint(checkpoint, device)
    sd = ckpt["model"]
    if any(k.startswith("hybrid_trunk.") for k in sd) or str(ckpt.get("backbone", "")).lower() == "hybrid":
        raise ValueError(
            "checkpoint 为已移除的 hybrid 主干，无法加载；请换用 ResNet 两阶段权重或回退到仍含 hybrid 的仓库版本。"
        )
    if any(k.startswith("xfm_trunk.") for k in sd) or str(ckpt.get("backbone", "")).lower() == "transformer":
        raise ValueError(
            "checkpoint 为 transformer 主干，本版本已改为仅 ResNet 塔；请换用 resnet 权重或使用旧分支。"
        )
    filters = int(ckpt.get("filters", _infer_filters_from_state(sd)))
    num_res = int(ckpt.get("num_res_layers", 0))
    if num_res <= 0:
        num_res = count_resnet_blocks_in_state(sd) or 10
    in_ch = int(
        in_channels
        if in_channels is not None
        else ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS))
    )
    model = SuccessorPolicy(
        num_res_layers=num_res,
        in_channels=in_ch,
        filters=filters,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    flist: dict[str, list[str]] = {
        "red": list(FEATURE_LIST["red"]),
        "black": list(FEATURE_LIST["black"]),
    }
    return model, flist
