"""PyTorch 合法着法后继局面打分网络（棋谱着法 CE）+ 红方胜负和三分类价值头（棋谱 RecordResult CE）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS, VALUE_LABEL_IGNORE


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
    对每个候选着法对应的后继局面平面独立前向，输出标量 logit（键名 ``score`` 以兼容旧 checkpoint）；
    另含共享 trunk 的价值头：在走棋**之前**的当前局面 ``x_current`` 上输出红方胜/和/负三个 logit。

    前向：
    - 仅 ``x``：返回 ``(B, K)`` 着法 logits（对弈/旧脚本）。
    - ``x`` 与 ``x_current``：返回 ``(policy_logits, value_logits)``，形状 ``(B, K)`` 与 ``(B, 3)``。
    """

    def __init__(
        self,
        num_res_layers: int = 10,
        in_channels: int | None = None,
        filters: int = 256,
    ) -> None:
        super().__init__()
        c = in_channels if in_channels is not None else POLICY_SELECT_IN_CHANNELS
        self.in_channels = c
        self.stem_conv = nn.Conv2d(c, filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_res_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score = nn.Linear(filters, 1)
        self.value_head = nn.Linear(filters, 3)

    def _trunk_flat(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """``x_nchw``: (N, C, H, W) -> (N, filters)。"""
        t = F.elu(self.stem_bn(self.stem_conv(x_nchw)))
        t = self.blocks(t)
        return self.pool(t).flatten(1)

    def forward(
        self, x: torch.Tensor, x_current: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        ``x``: (B, K, C, H, W) 后继局面。
        ``x_current`` 若有：``(B, C, H, W)`` 为走棋前当前局面，与 ``x`` 共用 stem+ResNet。
        """
        b, k, c, h, w = x.shape
        if x_current is None:
            feat = self._trunk_flat(x.reshape(b * k, c, h, w)).view(b, k, -1)
            return self.score(feat).squeeze(-1)
        t_succ = x.reshape(b * k, c, h, w)
        feat_all = self._trunk_flat(torch.cat([t_succ, x_current], dim=0))
        feat_p = feat_all[: b * k].view(b, k, -1)
        feat_v = feat_all[b * k :]
        return self.score(feat_p).squeeze(-1), self.value_head(feat_v)

    def predict_move_logits(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """x: (1, K, C, H, W)，返回 (K,) logits（无 softmax）。"""
        self.eval()
        with torch.no_grad():
            out = self.forward(x_nchw, None)
            assert isinstance(out, torch.Tensor)
            return out[0]


def batched_successors_nhwc_to_torch(
    x: np.ndarray,
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """(B, K, H, W, C) float32 numpy -> (B, K, C, H, W) tensor。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 1, 4, 2, 3)
    if pin_memory:
        t = t.pin_memory()
    return t.to(device, non_blocking=non_blocking)


def batched_current_nhwc_to_torch(
    x: np.ndarray,
    device: torch.device,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    """(B, H, W, C) float32 numpy -> (B, C, H, W) tensor（走棋前当前局面）。"""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(0, 3, 1, 2)
    if pin_memory:
        t = t.pin_memory()
    return t.to(device, non_blocking=non_blocking)


def nhwc_numpy_to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B,H,W,C) float32 numpy -> (B,C,H,W) tensor（保留供其它脚本使用）。"""
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.float().permute(0, 3, 1, 2).to(device)


def logits_as_red_preference(
    logits: torch.Tensor, red_to_move: torch.Tensor | bool
) -> torch.Tensor:
    """
    将网络 raw 输出解释为「对红方越有利则标量越大」：红方走棋侧不变，黑方走棋侧取负。
    ``red_to_move`` 为走棋**之前**是否轮到红方；批张量形状 ``(B,)``，与 ``logits`` 的 ``B`` 一致。
    """
    if isinstance(red_to_move, bool):
        return logits if red_to_move else -logits
    s = torch.where(red_to_move, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype).unsqueeze(1)
    return logits * s


def accuracy_from_logits_masked(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """仅在有效槽位上 argmax 与 target 比较。"""
    logits = logits.masked_fill(~mask, -1e9)
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()


def value_accuracy_ignore(
    logits_3: torch.Tensor, target: torch.Tensor, ignore_index: int = VALUE_LABEL_IGNORE
) -> torch.Tensor:
    """红方结果三分类准确率；``ignore_index`` 样本不计入分母。"""
    m = target != ignore_index
    if not m.any():
        return torch.tensor(0.0, device=logits_3.device, dtype=torch.float32)
    pred = logits_3[m].argmax(dim=1)
    return (pred == target[m]).float().mean()


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    """兼容 PyTorch 2.6+ 默认 weights_only 行为。"""
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)
