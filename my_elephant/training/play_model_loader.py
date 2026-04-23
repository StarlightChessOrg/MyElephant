"""从 checkpoint 构建对弈用 ``SuccessorPolicy``（供 ``play_policy_torch`` 与 HTTP 评估子进程复用）。"""

from __future__ import annotations

from pathlib import Path

import torch

from my_elephant.chess import FEATURE_LIST
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    count_hybrid_stem_res_blocks_in_state,
    count_transformer_encoder_layers_in_state,
    default_transformer_nhead,
    torch_load_checkpoint,
)


def _infer_filters_from_state(sd: dict) -> int:
    w = sd.get("hybrid_trunk.stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    w = sd.get("xfm_trunk.in_proj.weight")
    if w is not None:
        return int(w.shape[0])
    w = sd.get("stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    return 64


def load_successor_policy_for_play(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int | None = None,
) -> tuple[SuccessorPolicy, dict[str, list[str]]]:
    ckpt = torch_load_checkpoint(checkpoint, device)
    sd = ckpt["model"]
    backbone = str(ckpt.get("backbone", "")).lower()
    if backbone not in ("hybrid", "transformer", "resnet"):
        if any(k.startswith("hybrid_trunk.") for k in sd):
            backbone = "hybrid"
        elif any(k.startswith("xfm_trunk.") for k in sd):
            backbone = "transformer"
        else:
            backbone = "resnet"
    filters = int(ckpt.get("filters", _infer_filters_from_state(sd)))
    num_res = int(ckpt.get("num_res_layers", 0))
    if num_res <= 0 and backbone in ("transformer", "hybrid"):
        pref = (
            "hybrid_trunk.encoder.layers."
            if backbone == "hybrid"
            else "xfm_trunk.encoder.layers."
        )
        num_res = count_transformer_encoder_layers_in_state(sd, pref) or 4
    elif num_res <= 0:
        num_res = 4
    in_ch = int(
        ckpt.get("in_channels", ckpt.get("select_in_channels", in_channels or POLICY_SELECT_IN_CHANNELS))
    )
    stem_rb = ckpt.get("stem_res_blocks")
    if stem_rb is not None:
        stem_rb = int(stem_rb)
    elif backbone == "hybrid":
        stem_rb = max(0, count_hybrid_stem_res_blocks_in_state(sd))
    else:
        stem_rb = 2
    nhead = ckpt.get("nhead")
    dim_ff = ckpt.get("dim_feedforward")
    if backbone in ("transformer", "hybrid"):
        if nhead is None:
            nhead = default_transformer_nhead(filters)
        else:
            nhead = int(nhead)
        if dim_ff is not None:
            dim_ff = int(dim_ff)
    model = SuccessorPolicy(
        num_res_layers=num_res,
        in_channels=in_ch,
        filters=filters,
        backbone=backbone,
        stem_res_blocks=stem_rb,
        nhead=int(nhead) if backbone in ("transformer", "hybrid") and nhead is not None else nhead,
        dim_feedforward=dim_ff,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    flist: dict[str, list[str]] = {
        "red": list(FEATURE_LIST["red"]),
        "black": list(FEATURE_LIST["black"]),
    }
    return model, flist
