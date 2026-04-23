"""PUCT 式 MCTS，配合策略（起点/落点）+ 价值网络选着。

并行：默认按 CPU 逻辑核心数启动工作线程，在持锁的快速选路阶段之外并发调用 ``evaluator``（网络前向），
以缩短墙钟；搜索树仍在单进程内共享（与 CUDA 模型兼容）。真多进程需每进程独立模型与根并行再合并根统计，与本实现不同。

虚拟损失：沿路径选边时对 ``(parent, action)`` 增加 ``parent.in_flight[action]``；PUCT 中
``Q ≈ (W - virtual_loss * inflight) / (N + inflight)``；备份后递减，减轻多线程挤占同一路径。
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parallel_workers: int
    virtual_loss: float


class _MCTSNode:
    __slots__ = (
        "gp",
        "parent",
        "move_from_parent",
        "children",
        "P",
        "N",
        "W",
        "expanded",
        "in_flight",
    )

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
        self.in_flight: dict[str, int] = {}


def _puct_select(node: _MCTSNode, c_puct: float, virtual_loss: float) -> str:
    total = 0
    for m in node.P:
        total += node.N.get(m, 0) + node.in_flight.get(m, 0)
    sqrt_n = math.sqrt(max(1, total))
    best_m, best_u = None, -1e18
    for m in node.P:
        n = node.N.get(m, 0)
        ifl = node.in_flight.get(m, 0)
        w = node.W.get(m, 0.0)
        denom = max(1e-8, n + ifl)
        q = (w - virtual_loss * ifl) / denom
        u = q + c_puct * node.P[m] * sqrt_n / (1.0 + n + ifl)
        if u > best_u:
            best_u, best_m = u, m
    assert best_m is not None
    return best_m


def _apply_expand(node: _MCTSNode, legals_s: list[str], priors: np.ndarray) -> None:
    """将 ``evaluator`` 结果写入节点（不再次调用网络）。"""
    if not legals_s:
        return
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


def _expand(node: _MCTSNode, evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]]) -> float:
    legals_s, priors, v = evaluator(node.gp)
    if not legals_s:
        return _terminal_outcome(node.gp) or 0.0
    _apply_expand(node, legals_s, priors)
    return v


def _backup(path: list[tuple[_MCTSNode, str]], v: float) -> None:
    """``v`` 为叶子处「轮到走棋方」的估值；沿 ``path`` 向上回传。"""
    cur = v
    for par, a in reversed(path):
        par.N[a] = par.N.get(a, 0) + 1
        par.W[a] = par.W.get(a, 0.0) - cur
        cur = -cur


def _release_inflight_path(path: list[tuple[_MCTSNode, str]]) -> None:
    for par, a in path:
        c = par.in_flight.get(a, 0) - 1
        if c <= 0:
            par.in_flight.pop(a, None)
        else:
            par.in_flight[a] = c


def _single_playout(
    root: _MCTSNode,
    evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]],
    c_puct: float,
    virtual_loss: float,
    lock: threading.RLock,
    counters: dict[str, int],
    t0: float,
    max_seconds: float | None,
    n_simulations: int,
) -> None:
    """一次模拟（另一线程抢先展开同一叶子时会重试选路）。"""
    while True:
        path: list[tuple[_MCTSNode, str]] = []
        expand_node: _MCTSNode | None = None
        gp_eval: GamePlay | None = None

        with lock:
            if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                return
            if counters["n_completed"] >= n_simulations:
                return

            node = root
            while True:
                out = _terminal_outcome(node.gp)
                if out is not None:
                    _backup(path, out)
                    _release_inflight_path(path)
                    counters["n_completed"] += 1
                    return

                if not node.expanded:
                    expand_node = node
                    gp_eval = copy_gameplay(node.gp)
                    break

                a = _puct_select(node, c_puct, virtual_loss)
                node.in_flight[a] = node.in_flight.get(a, 0) + 1
                path.append((node, a))
                node = node.children[a]

        assert expand_node is not None and gp_eval is not None
        legals_s, priors, v = evaluator(gp_eval)

        with lock:
            counters["n_expansions"] += 1
            if not legals_s:
                v2 = _terminal_outcome(expand_node.gp) or 0.0
                _backup(path, v2)
                _release_inflight_path(path)
                counters["n_completed"] += 1
                return

            expanded_here = not expand_node.expanded
            if expanded_here:
                _apply_expand(expand_node, legals_s, priors)

            if expanded_here:
                _backup(path, v)
                _release_inflight_path(path)
                counters["n_completed"] += 1
                return

            _release_inflight_path(path)
            continue


def _mcts_search_sequential(
    root_gp: GamePlay,
    evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]],
    n_simulations: int,
    c_puct: float,
    max_seconds: float | None,
    t0: float,
) -> tuple[_MCTSNode, int, int, str]:
    """单线程原版循环；返回 (root, n_playouts, n_expansions, stopped_by)。"""
    root = _MCTSNode(copy_gameplay(root_gp))
    if _terminal_outcome(root.gp) is not None:
        raise RuntimeError("根节点已终局")

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
            a = _puct_select(node, c_puct, 0.0)
            path.append((node, a))
            node = node.children[a]

    return root, n_playouts, n_expansions, stopped_by


def mcts_search(
    root_gp: GamePlay,
    evaluator: Callable[[GamePlay], tuple[list[str], np.ndarray, float]],
    n_simulations: int = 256,
    c_puct: float = 1.5,
    *,
    max_seconds: float | None = None,
    virtual_loss: float = 3.0,
    n_workers: int | None = None,
) -> tuple[str, MCTSSearchStats]:
    """
    从 ``root_gp`` 当前局面出发做 MCTS，返回 ``(走法, 统计)``。

    ``max_seconds``：每条模拟开始前检查；``None`` 表示不限时。

    ``n_workers``：并行工作线程数，默认 ``os.cpu_count()``；为 ``1`` 时用单线程路径。
    评估器在锁外调用；树更新在锁内。与 CUDA 共用单模型时采用多线程而非多进程。
    """
    if _terminal_outcome(copy_gameplay(root_gp)) is not None:
        raise RuntimeError("根节点已终局")

    t0 = time.perf_counter()
    nw = 1 if n_workers is not None and n_workers <= 1 else (n_workers or (os.cpu_count() or 1))
    nw = max(1, min(nw, 64, max(1, n_simulations)))

    if nw <= 1:
        root, n_playouts, n_expansions, stopped_by = _mcts_search_sequential(
            root_gp, evaluator, n_simulations, c_puct, max_seconds, t0
        )
        vl_report = 0.0
    else:
        root = _MCTSNode(copy_gameplay(root_gp))
        lock = threading.RLock()
        counters = {"n_completed": 0, "n_expansions": 0}
        vl_report = float(virtual_loss)

        def worker() -> None:
            while True:
                with lock:
                    if counters["n_completed"] >= n_simulations:
                        return
                    if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
                        return
                _single_playout(
                    root,
                    evaluator,
                    c_puct,
                    virtual_loss,
                    lock,
                    counters,
                    t0,
                    max_seconds,
                    n_simulations,
                )

        with ThreadPoolExecutor(max_workers=nw) as ex:
            futures = [ex.submit(worker) for _ in range(nw)]
            for f in as_completed(futures):
                f.result()

        n_playouts = counters["n_completed"]
        n_expansions = counters["n_expansions"]
        stopped_by = "simulations"
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            stopped_by = "time"
        elif n_playouts >= n_simulations:
            stopped_by = "simulations"

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
        parallel_workers=nw,
        virtual_loss=vl_report,
    )
    return best, stats
