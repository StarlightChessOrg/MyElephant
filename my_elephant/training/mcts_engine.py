"""PUCT 式 MCTS，配合策略（起点/落点）+ 价值网络选着。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from my_elephant.chess.board_utils import chess_board_from_base
from my_elephant.chess.session import GamePlay


def copy_gameplay(g: GamePlay) -> GamePlay:
    o = GamePlay.__new__(GamePlay)
    o.bb = g.bb.copy()
    o.red = g.red
    return o


def _terminal_outcome(gp: GamePlay) -> float | None:
    """若无合法着：将死则轮到方 -1，否则和 0；有合法着返回 None。"""
    if gp.legal_moves_iccs():
        return None
    cb = chess_board_from_base(gp.bb)
    if cb.is_checkmate():
        return -1.0
    return 0.0


@dataclass(frozen=True)
class MCTSSearchStats:
    """
    一次 MCTS 结束后的简要统计（便于 UI / 日志）。

    ``stopped_by``：``"simulations"`` 表示因达到 ``n_simulations`` 退出循环；``"time"`` 表示因时间上限退出。
    ``n_expansions``：调用 ``evaluator``（网络展开叶子）的次数。
    ``root_total_visits``：根各着法访问计数之和（与 ``n_playouts`` 在根处通常接近）。
    """

    best_move: str
    n_playouts: int
    n_expansions: int
    root_total_visits: int
    elapsed_seconds: float
    stopped_by: str
    requested_simulations: int
    requested_max_seconds: float | None


class _MCTSNode:
    __slots__ = ("gp", "parent", "move_from_parent", "children", "P", "N", "W", "expanded")

    def __init__(
        self,
        gp: GamePlay,
        parent: _MCTSNode | None = None,
        move_from_parent: str | None = None,
    ) -> None:
        self.gp = gp
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children: dict[str, _MCTSNode] = {}
        self.P: dict[str, float] = {}
        self.N: dict[str, int] = {}
        self.W: dict[str, float] = {}
        self.expanded = False


def _puct_select(node: _MCTSNode, c_puct: float) -> str:
    total = sum(node.N.get(m, 0) for m in node.P)
    sqrt_n = math.sqrt(max(1, total))
    best_m, best_u = None, -1e18
    for m in node.P:
        n = node.N.get(m, 0)
        w = node.W.get(m, 0.0)
        q = w / n if n > 0 else 0.0
        u = q + c_puct * node.P[m] * sqrt_n / (1 + n)
        if u > best_u:
            best_u = u
            best_m = m
    assert best_m is not None
    return best_m


def _expand(node: _MCTSNode, evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]]) -> float:
    legals_s, priors, v = evaluator(node.gp)
    if not legals_s:
        return _terminal_outcome(node.gp) or 0.0
    s = float(np.sum(priors))
    if s <= 0:
        p = {m: 1.0 / len(legals_s) for m in legals_s}
    else:
        p = {m: float(priors[i]) / s for i, m in enumerate(legals_s)}
    node.P = p
    for m in legals_s:
        child = copy_gameplay(node.gp)
        child.make_move(m)
        node.children[m] = _MCTSNode(child, parent=node, move_from_parent=m)
    node.expanded = True
    return v


def _backup(path: list[tuple[_MCTSNode, str]], v: float) -> None:
    """``v`` 为叶子处「轮到走棋方」的估值；沿 ``path`` 向上回传。"""
    cur = v
    for par, a in reversed(path):
        par.N[a] = par.N.get(a, 0) + 1
        par.W[a] = par.W.get(a, 0.0) - cur
        cur = -cur


def mcts_search(
    root_gp: GamePlay,
    evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]],
    n_simulations: int = 256,
    c_puct: float = 1.5,
    *,
    max_seconds: float | None = None,
) -> tuple[str, MCTSSearchStats]:
    """
    从 ``root_gp`` 当前局面出发做 MCTS，返回 ``(走法, 统计)``。
    在每条 simulation **开始前**检查时间；``max_seconds`` 为 ``None`` 时不限时。
    ``evaluator(gp)`` 须返回 ``(合法走法 str 列表, prior 向量与列表对齐, 轮到方价值 v)``。
    """
    root = _MCTSNode(copy_gameplay(root_gp))
    if _terminal_outcome(root.gp) is not None:
        raise RuntimeError("根节点已终局")

    t0 = time.perf_counter()
    n_playouts = 0
    n_expansions = 0
    stopped_by = "simulations"

    while n_playouts < n_simulations:
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            stopped_by = "time"
            break
        n_playouts += 1
        node = root
        path: list[tuple[_MCTSNode, str]] = []
        while True:
            out = _terminal_outcome(node.gp)
            if out is not None:
                _backup(path, out)
                break
            if not node.expanded:
                v = _expand(node, evaluator)
                n_expansions += 1
                _backup(path, v)
                break
            a = _puct_select(node, c_puct)
            path.append((node, a))
            node = node.children[a]

    if not root.P:
        raise RuntimeError("MCTS 未展开")
    best = max(root.P.keys(), key=lambda m: root.N.get(m, 0))
    elapsed = time.perf_counter() - t0
    root_visits = int(sum(root.N.get(m, 0) for m in root.P))
    stats = MCTSSearchStats(
        best_move=best,
        n_playouts=n_playouts,
        n_expansions=n_expansions,
        root_total_visits=root_visits,
        elapsed_seconds=elapsed,
        stopped_by=stopped_by,
        requested_simulations=n_simulations,
        requested_max_seconds=max_seconds,
    )
    return best, stats
