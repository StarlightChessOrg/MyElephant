"""PyTorch 两阶段策略（起点格 + 落点格）+ 红方胜负和价值头。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cchess.piece import ChessSide

from my_elephant.chess.features import encode_model_planes
from my_elephant.chess.rationale import POLICY_GRID_NUMEL, POLICY_SELECT_IN_CHANNELS, VALUE_LABEL_IGNORE
from my_elephant.chess.session import GamePlay


class ResBlock(nn.Module):
    """两个 3x3 卷积 + BN，在第二支末尾与输入相加后 ELU。"""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.elu(out)
        return out


class SuccessorPolicy(nn.Module):
    """
    共享 trunk 读走棋前当前局面 ``(B,C,10,9)``：
    - ``head_src``：90 维，选起点格（ICCS 展平 ``y*9+x``）；
    - ``head_dst``：在 teacher/推理给定起点 one-hot 拼接后，90 维选落点格；
    - ``value_head``：红方胜/和/负三分类。
    """

    def __init__(
        self,
        num_res_layers: int = 10,
        in_channels: int | None = None,
        filters: int = 256,
        grid: int = POLICY_GRID_NUMEL,
    ) -> None:
        super().__init__()
        c = in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS
        self.in_channels = c
        self.grid = grid
        self.stem_conv = nn.Conv2d(c, filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_res_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_src = nn.Linear(filters, grid)
        self.head_dst = nn.Linear(filters + grid, grid)
        self.value_head = nn.Linear(filters, 3)
        self.filters = filters

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        t = self.blocks(t)
        return self.pool(t).flatten(1)

    def forward(
        self, x_cur: torch.Tensor, src_one_hot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``x_cur``: (B, C, H, W) 走棋前当前局面。
        ``src_one_hot``: (B, 90) 训练时为真起点 one-hot；推理第二步时为上一步选中的起点 one-hot。
        返回 ``(logits_src, logits_dst, logits_val)``，各为 ``(B,90)``、``(B,90)``、``(B,3)``。
        """
        feat = self._trunk_flat(x_cur)
        logits_src = self.head_src(feat)
        logits_dst = self.head_dst(torch.cat([feat, src_one_hot], dim=1))
        logits_val = self.value_head(feat)
        return logits_src, logits_dst, logits_val


def batched_current_nhwc_to_torch(
    x: np.ndarray,
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """(B, H, W, C) float32 numpy -> (B, C, H, W) tensor。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 3, 1, 2)
    if pin_memory:
        t = t.pin_memory()
    return t.to(device, non_blocking=non_blocking)


def batched_successors_nhwc_to_torch(
    x: np.ndarray,
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """保留 API：旧版后继平面 batch；当前训练不再使用。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 1, 4, 2, 3)
    if pin_memory:
        t = t.pin_memory()
    return t.to(device, non_blocking=non_blocking)


def nhwc_numpy_to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.float().permute(0, 3, 1, 2).to(device)


def logits_as_red_preference(
    logits: torch.Tensor, red_to_move: torch.Tensor | bool
) -> torch.Tensor:
    """将标量 logits 解释为「对红方越有利越大」（用于价值等）；``logits`` 形状含 batch 维时最后一维为 1 或标量。"""
    if isinstance(red_to_move, bool):
        return logits if red_to_move else -logits
    s = torch.where(red_to_move, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
    while s.dim() < logits.dim():
        s = s.unsqueeze(-1)
    return logits * s


def accuracy_from_logits_masked(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e9)
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()


def joint_move_accuracy(
    logits_s: torch.Tensor,
    logits_d: torch.Tensor,
    msrc: torch.Tensor,
    mdst: torch.Tensor,
    tgt_s: torch.Tensor,
    tgt_d: torch.Tensor,
) -> torch.Tensor:
    """两阶段均预测正确（同一 batch 内）的比例。"""
    ps = logits_s.masked_fill(~msrc, -1e9).argmax(dim=1)
    pd = logits_d.masked_fill(~mdst, -1e9).argmax(dim=1)
    return ((ps == tgt_s) & (pd == tgt_d)).float().mean()


def value_accuracy_ignore(
    logits_3: torch.Tensor, target: torch.Tensor, ignore_index: int = VALUE_LABEL_IGNORE
) -> torch.Tensor:
    m = target != ignore_index
    if not m.any():
        return torch.tensor(0.0, device=logits_3.device, dtype=torch.float32)
    pred = logits_3[m].argmax(dim=1)
    return (pred == target[m]).float().mean()


def _encode_gameplay_current_nchw(
    gameplay: GamePlay, flist: dict[str, list[str]], device: torch.device
) -> torch.Tensor:
    raw = np.asarray(gameplay.bb._board[::-1])
    red_to = gameplay.bb.move_side is not ChessSide.BLACK
    cur_chw = encode_model_planes(raw, red_to, gameplay.bb, flist)
    cur_hwc = np.transpose(cur_chw, (1, 2, 0))
    t = (
        torch.from_numpy(np.ascontiguousarray(np.expand_dims(cur_hwc, 0)))
        .float()
        .permute(0, 3, 1, 2)
        .to(device)
    )
    return t


@torch.no_grad()
def infer_greedy_move_string(
    gameplay: GamePlay,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> str:
    """单步贪心：先 argmax 合法起点，再在该起点下 argmax 合法落点，返回 ``x1y1-x2y2``。"""
    legals_t = sorted(gameplay.legal_moves_iccs())
    if not legals_t:
        raise RuntimeError("无合法着法")
    x_cur = _encode_gameplay_current_nchw(gameplay, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
    ls = model.head_src(feat)[0]
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, _, _ in legals_t:
        src_mask[y1 * 9 + x1] = True
    ls = ls.masked_fill(~src_mask, -1e9)
    src_i = int(torch.argmax(ls).item())
    oh = torch.zeros(1, POLICY_GRID_NUMEL, device=device, dtype=feat.dtype)
    oh[0, src_i] = 1.0
    ld = model.head_dst(torch.cat([feat, oh], dim=1))[0]
    sx, sy = src_i % 9, src_i // 9
    dst_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, x2, y2 in legals_t:
        if (x1, y1) == (sx, sy):
            dst_mask[y2 * 9 + x2] = True
    ld = ld.masked_fill(~dst_mask, -1e9)
    dst_i = int(torch.argmax(ld).item())
    dx, dy = dst_i % 9, dst_i // 9
    return f"{sx}{sy}-{dx}{dy}"


@torch.no_grad()
def eval_policy_value_at_root(
    gameplay: GamePlay,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> tuple[list[str], np.ndarray, float]:
    """
    MCTS 用：对当前局面枚举合法着法，用分解式 ``P(着)=P(起点)P(落点|起点)`` 得到与 ``legals`` 对齐的 prior，
    并给出轮到方价值标量（由红方三分类经视角换算）。
    """
    legals_t = sorted(gameplay.legal_moves_iccs())
    if not legals_t:
        return [], np.zeros(0, dtype=np.float64), 0.0
    legals_s = [f"{a}{b}-{c}{d}" for (a, b, c, d) in legals_t]
    x_cur = _encode_gameplay_current_nchw(gameplay, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
    ls = model.head_src(feat)[0]
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, _, _ in legals_t:
        src_mask[y1 * 9 + x1] = True
    p_src = torch.softmax(ls.masked_fill(~src_mask, -1e9), dim=0)
    origins: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x1, y1, _, _ in legals_t:
        if (x1, y1) not in seen:
            seen.add((x1, y1))
            origins.append((x1, y1))
    origins.sort()
    p_dst_given: dict[tuple[int, int], torch.Tensor] = {}
    for ox, oy in origins:
        oh = torch.zeros(1, POLICY_GRID_NUMEL, device=device, dtype=feat.dtype)
        oh[0, oy * 9 + ox] = 1.0
        ld = model.head_dst(torch.cat([feat, oh], dim=1))[0]
        dm = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        for x1, y1, x2, y2 in legals_t:
            if (x1, y1) == (ox, oy):
                dm[y2 * 9 + x2] = True
        p_dst_given[(ox, oy)] = torch.softmax(ld.masked_fill(~dm, -1e9), dim=0)
    pri: list[float] = []
    for x1, y1, x2, y2 in legals_t:
        i_s = y1 * 9 + x1
        i_d = y2 * 9 + x2
        pri.append(float((p_src[i_s] * p_dst_given[(x1, y1)][i_d]).item()))
    pri_arr = np.asarray(pri, dtype=np.float64)
    s = float(pri_arr.sum())
    if s > 0:
        pri_arr /= s
    logits_v = model.value_head(feat)[0]
    pv = torch.softmax(logits_v.float(), dim=0).cpu().numpy()
    v_red = float(pv[0] - pv[2])
    red = gameplay.get_side() == "red"
    v = v_red if red else -v_red
    return legals_s, pri_arr, v


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)
