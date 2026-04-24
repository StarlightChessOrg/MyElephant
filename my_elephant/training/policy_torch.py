"""PyTorch 两阶段策略（起点格 + 落点格）+ 红方胜负和价值头。"""
from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_elephant.chess.features import encode_model_planes
from my_elephant.chess.rationale import POLICY_GRID_NUMEL, POLICY_SELECT_IN_CHANNELS, VALUE_LABEL_IGNORE
from my_elephant.chess.session import GamePlay


def count_resnet_blocks_in_state(sd: dict, prefix: str = "blocks.") -> int:
    """从 ``state_dict`` 键名推断 ``Sequential`` 残差块个数（与 ``stem_conv`` + ``blocks`` + ``pool`` 塔一致）。"""
    mx = -1
    for k in sd:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        lead = rest.split(".", 1)[0]
        if lead.isdigit():
            mx = max(mx, int(lead))
    return mx + 1 if mx >= 0 else 0


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
    - ``value_head``：**行棋方**胜/和/负三分类 logit（与棋谱经 ``stm_outcome_class_from_red_outcome`` 对齐）。

    卷积塔与 ``StarlightChessOrg/MyElephant`` 提交 ``91e27b25`` 中 ``SuccessorPolicy`` 一致：
    ``3×3 stem`` + ``num_res_layers`` 个 ``ResBlock`` + 全局平均池化；在此之上接两阶段策略头与价值头。
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
        self.filters = filters
        self.stem_conv = nn.Conv2d(c, filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(filters)
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_res_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head_src = nn.Linear(filters, grid)
        self.head_dst = nn.Linear(filters + grid, grid)
        self.value_head = nn.Linear(filters, 3)

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
    # 固定红方物理视角；行棋方不进平面（由交互/规则隐含）
    cur_chw = encode_model_planes(
        raw,
        gameplay.red,
        gameplay.bb,
        flist,
        last_move=gameplay.last_move_iccs,
    )
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


def _policy_temperature_scalar(policy_temperature: float) -> float:
    """合法正温度；非法或非正时退化为 1.0（与旧版 softmax 一致）。"""
    t = float(policy_temperature)
    if not (t > 0.0) or math.isnan(t) or math.isinf(t):
        return 1.0
    return t


@torch.no_grad()
def _root_prior_value_from_hidden_row(
    feat: torch.Tensor,
    ls_logits: torch.Tensor,
    legals_t: list[tuple[int, int, int, int]],
    model: SuccessorPolicy,
    device: torch.device,
    *,
    policy_temperature: float = 1.0,
) -> tuple[list[str], np.ndarray, float]:
    """单局面：已有 ``trunk`` 隐层与 ``head_src`` 一行 logits，完成落点头与价值，返回 ``(legals_s, pri_arr, v)``。"""
    T = _policy_temperature_scalar(policy_temperature)
    legals_s = [f"{a}{b}-{c}{d}" for (a, b, c, d) in legals_t]
    src_mask = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
    for x1, y1, _, _ in legals_t:
        src_mask[y1 * 9 + x1] = True
    p_src = torch.softmax((ls_logits / T).masked_fill(~src_mask, -1e9), dim=0)
    origins: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x1, y1, _, _ in legals_t:
        if (x1, y1) not in seen:
            seen.add((x1, y1))
            origins.append((x1, y1))
    origins.sort()
    k = len(origins)
    oh_batch = torch.zeros(k, POLICY_GRID_NUMEL, device=device, dtype=feat.dtype)
    for idx, (ox, oy) in enumerate(origins):
        oh_batch[idx, oy * 9 + ox] = 1.0
    feat_rep = feat.expand(k, -1)
    ld_batch = model.head_dst(torch.cat([feat_rep, oh_batch], dim=1))
    p_dst_given: dict[tuple[int, int], torch.Tensor] = {}
    for idx, (ox, oy) in enumerate(origins):
        ld = ld_batch[idx]
        dm = torch.zeros(POLICY_GRID_NUMEL, dtype=torch.bool, device=device)
        for x1, y1, x2, y2 in legals_t:
            if (x1, y1) == (ox, oy):
                dm[y2 * 9 + x2] = True
        p_dst_given[(ox, oy)] = torch.softmax((ld / T).masked_fill(~dm, -1e9), dim=0)
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
    v = float(pv[0] - pv[2])
    return legals_s, pri_arr, v


@torch.no_grad()
def batched_eval_policy_value_at_root(
    gameplays: list[GamePlay],
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
) -> list[tuple[list[str], np.ndarray, float]]:
    """
    多根局面：共享一次 ``trunk`` + ``head_src`` 批前向，再逐局完成 ``head_dst`` 与价值（各局合法起点数不同，落点头仍按局批）。
    与 ``eval_policy_value_at_root`` 数值一致；无合法着法的局面不占用 GPU。

    ``policy_temperature``：对 ``head_src`` / ``head_dst`` 的 logits 做 ``softmax(logits/T)``；``T>1`` 先验更平、``T<1`` 更尖（默认 ``1`` 与旧行为一致）。
    """
    n = len(gameplays)
    if n == 0:
        return []
    legals_per: list[list[tuple[int, int, int, int]]] = []
    nonempty: list[int] = []
    for i, gp in enumerate(gameplays):
        lt = sorted(gp.legal_moves_iccs())
        legals_per.append(lt)
        if lt:
            nonempty.append(i)
    out: list[tuple[list[str], np.ndarray, float] | None] = [None] * n
    for i in range(n):
        if not legals_per[i]:
            out[i] = ([], np.zeros(0, dtype=np.float64), 0.0)
    if not nonempty:
        return [out[i] for i in range(n)]  # type: ignore[list-item]
    model.eval()
    xb = torch.cat(
        [_encode_gameplay_current_nchw(gameplays[j], flist, device) for j in nonempty],
        dim=0,
    )
    feat_b = model._trunk_flat(xb)
    ls_b = model.head_src(feat_b)
    for bi, j in enumerate(nonempty):
        feat_row = feat_b[bi : bi + 1]
        ls_row = ls_b[bi]
        out[j] = _root_prior_value_from_hidden_row(
            feat_row, ls_row, legals_per[j], model, device, policy_temperature=policy_temperature
        )
    return [out[i] for i in range(n)]  # type: ignore[list-item]


@torch.no_grad()
def eval_policy_value_at_root(
    gameplay: GamePlay,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    policy_temperature: float = 1.0,
) -> tuple[list[str], np.ndarray, float]:
    """
    MCTS 用：对当前局面枚举合法着法，用分解式 ``P(着)=P(起点)P(落点|起点)`` 得到与 ``legals`` 对齐的 prior，
    并给出轮到方价值标量（由**行棋方**三分类 ``P(胜)-P(负)``，无需再按红黑翻转）。
    """
    return batched_eval_policy_value_at_root(
        [gameplay], model, device, flist, policy_temperature=policy_temperature
    )[0]


class QueuedBatchedRootEvaluator:
    """
    单消费线程合并多路 MCTS 的根评估请求，在 CUDA 上对 ``trunk``+``head_src`` 做批前向，避免多线程并行调同一模型。
    调用方在 ``eval_sync`` 上阻塞；须在线程池结束后再 ``close``（见对弈里 shutdown 顺序）。
    """

    _SENTINEL = object()

    def __init__(
        self,
        model: SuccessorPolicy,
        device: torch.device,
        flist: dict[str, list[str]],
        *,
        max_batch: int = 16,
        max_wait_s: float = 0.002,
        policy_temperature: float = 1.0,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("QueuedBatchedRootEvaluator 仅用于 CUDA 设备")
        self._model = model
        self._device = device
        self._flist = flist
        self._policy_temperature = float(policy_temperature)
        self._max_batch = max(1, int(max_batch))
        self._max_wait_s = max(0.0, float(max_wait_s))
        self._in_q: queue.Queue[tuple[GamePlay, queue.Queue] | object] = queue.Queue()
        self._closed = False
        self._thr = threading.Thread(target=self._consume_loop, name="batched-root-eval", daemon=True)
        self._thr.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._in_q.put(self._SENTINEL)

    def eval_sync(self, gp: GamePlay) -> tuple[list[str], np.ndarray, float]:
        if self._closed:
            raise RuntimeError("QueuedBatchedRootEvaluator 已关闭")
        out_q: queue.Queue[object] = queue.Queue(maxsize=1)
        self._in_q.put((gp, out_q))
        result = out_q.get()
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    def _consume_loop(self) -> None:
        while True:
            first = self._in_q.get()
            if first is self._SENTINEL:
                break
            batch: list[tuple[GamePlay, queue.Queue]] = [first]  # type: ignore[list-item]
            deadline = time.perf_counter() + self._max_wait_s
            while len(batch) < self._max_batch:
                wait = deadline - time.perf_counter()
                if wait <= 0:
                    break
                try:
                    nxt = self._in_q.get(timeout=wait)
                except queue.Empty:
                    break
                if nxt is self._SENTINEL:
                    self._in_q.put(nxt)
                    break
                batch.append(nxt)  # type: ignore[arg-type]
            gps = [b[0] for b in batch]
            try:
                outs = batched_eval_policy_value_at_root(
                    gps,
                    self._model,
                    self._device,
                    self._flist,
                    policy_temperature=self._policy_temperature,
                )
            except Exception as e:
                for _, rq in batch:
                    rq.put(e)
                continue
            for (_, rq), o in zip(batch, outs):
                rq.put(o)


@torch.no_grad()
def eval_value_side_to_move(
    gameplay: GamePlay,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> float:
    """轮到方视角的价值标量（约 ``[-1,1]``）：无着时将死为 ``-1``、否则和 ``0``；否则为价值头 ``P(行棋方胜)-P(行棋方负)``。"""
    if not gameplay.legal_moves_iccs():
        from my_elephant.chess.board_utils import chess_board_from_base

        cb = chess_board_from_base(gameplay.bb)
        if cb.is_checkmate():
            return -1.0
        return 0.0
    x_cur = _encode_gameplay_current_nchw(gameplay, flist, device)
    model.eval()
    feat = model._trunk_flat(x_cur)
    logits_v = model.value_head(feat)[0]
    pv = torch.softmax(logits_v.float(), dim=0)
    return float((pv[0] - pv[2]).item())


@torch.no_grad()
def infer_1ply_value_prior_move(
    gameplay: GamePlay,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
    *,
    prior_weight: float = 1.0,
    value_weight: float = 1.0,
    prior_eps: float = 1e-8,
    policy_temperature: float = 1.0,
) -> str:
    """
    根节点单层搜索：枚举全部合法着，用 ``eval_policy_value_at_root`` 的 prior ``P(a)``，
    对每个后继局面 ``s'`` 用价值头得到轮到方 ``V(s')``，走子方打分
    ``prior_weight * log(P(a)+eps) + value_weight * (-V(s'))``（零和近似），取 argmax。
    """
    from my_elephant.chess.board_utils import chess_board_from_base
    from my_elephant.training.mcts_engine import copy_gameplay

    legals_s, pri_arr, _ = eval_policy_value_at_root(
        gameplay, model, device, flist, policy_temperature=policy_temperature
    )
    if not legals_s:
        raise RuntimeError("无合法着法")
    n = len(legals_s)
    pri_use = np.asarray(pri_arr, dtype=np.float64).copy()
    if float(pri_use.sum()) <= 0.0:
        pri_use.fill(1.0 / n)

    q_root = np.zeros(n, dtype=np.float64)
    batch_gps: list[GamePlay] = []
    batch_orig: list[int] = []

    for i, m in enumerate(legals_s):
        g2 = copy_gameplay(gameplay)
        g2.make_move(m)
        if not g2.legal_moves_iccs():
            cb = chess_board_from_base(g2.bb)
            v_stm = -1.0 if cb.is_checkmate() else 0.0
            q_root[i] = -float(v_stm)
        else:
            batch_orig.append(i)
            batch_gps.append(g2)

    if batch_gps:
        xb = torch.cat(
            [_encode_gameplay_current_nchw(g, flist, device) for g in batch_gps],
            dim=0,
        )
        model.eval()
        feat_b = model._trunk_flat(xb)
        logits_v = model.value_head(feat_b)
        pv = torch.softmax(logits_v.float(), dim=1)
        v_stm_child = pv[:, 0] - pv[:, 2]
        for j, orig_i in enumerate(batch_orig):
            q_root[orig_i] = -float(v_stm_child[j].item())

    best_i = 0
    best_sc = -1e18
    for i in range(n):
        sc = prior_weight * math.log(pri_use[i] + prior_eps) + value_weight * float(q_root[i])
        if sc > best_sc:
            best_sc, best_i = sc, i
    return legals_s[best_i]


def torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict:
    p = Path(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(p, map_location=map_location)
