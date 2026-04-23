"""PUCT 式 MCTS，配合策略（起点/落点）+ 价值网络选着。"""

from __future__ import annotations

import math
from collections.abc import Callable

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
) -> str:
    """
    从 ``root_gp`` 当前局面出发做 MCTS，返回根下访问次数最多的走法字符串（如 ``77-47``）。
    ``evaluator(gp)`` 须返回 ``(合法走法 str 列表, prior 向量与列表对齐, 轮到方价值 v)``。
    """
    root = _MCTSNode(copy_gameplay(root_gp))
    if _terminal_outcome(root.gp) is not None:
        raise RuntimeError("根节点已终局")

    for _ in range(n_simulations):
        node = root
        path: list[tuple[_MCTSNode, str]] = []
        while True:
            out = _terminal_outcome(node.gp)
            if out is not None:
                _backup(path, out)
                break
            if not node.expanded:
                v = _expand(node, evaluator)
                _backup(path, v)
                break
            a = _puct_select(node, c_puct)
            path.append((node, a))
            node = node.children[a]

    if not root.P:
        raise RuntimeError("MCTS 未展开")
    return max(root.P.keys(), key=lambda m: root.N.get(m, 0))
