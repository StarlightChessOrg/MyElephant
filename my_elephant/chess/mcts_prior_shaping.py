"""
MCTS 先验整形：将军 > 吃子（MVV/LVA）> 其余，在网络 prior 上乘性加权后归一化。

用于提高对「换子」「将军」的搜索敏感度，不改变合法着集合，仅重排 / 放大 PUCT 中的先验项。
"""

from __future__ import annotations

import math

import numpy as np

from my_elephant.chess.board_utils import chess_board_from_base
from my_elephant.chess.features import parse_move_squares
from my_elephant.chess.session import GamePlay


def _copy_gp(g: GamePlay) -> GamePlay:
    o = GamePlay.__new__(GamePlay)
    o.bb = g.bb.copy()
    o.red = g.red
    return o


# 子力价值（用于 MVV/LVA；将帅极大以免被轻易「换」掉）
_PIECE_VALUE: dict[str, int] = {
    "k": 20000,
    "K": 20000,
    "r": 950,
    "R": 950,
    "n": 480,
    "N": 480,
    "c": 480,
    "C": 480,
    "b": 210,
    "B": 210,
    "a": 210,
    "A": 210,
    "p": 110,
    "P": 110,
}


def _piece_value(fench: str | None) -> int:
    if not fench:
        return 0
    return int(_PIECE_VALUE.get(fench, 120))


def _is_red_piece(ch: str) -> bool:
    return len(ch) == 1 and ch.isupper()


def _is_black_piece(ch: str) -> bool:
    return len(ch) == 1 and ch.islower()


def _is_enemy(fench: str | None, mover_is_red: bool) -> bool:
    if not fench:
        return False
    if mover_is_red:
        return _is_black_piece(fench)
    return _is_red_piece(fench)


def move_is_capture(gp: GamePlay, mv: str) -> bool:
    x1, y1, x2, y2 = parse_move_squares(mv)
    bb = gp.bb
    dst = bb._board[y2][x2]
    return _is_enemy(dst, gp.red)


def move_gives_check(gp: GamePlay, mv: str) -> bool:
    g2 = _copy_gp(gp)
    g2.make_move(mv)
    cb = chess_board_from_base(g2.bb)
    return int(cb.is_checked()) > 0


def _mvv_lva_capture_multiplier(gp: GamePlay, mv: str, *, strength: float) -> float:
    """吃子时：高价值子被吃、低价值子去吃 → 乘子更大。"""
    x1, y1, x2, y2 = parse_move_squares(mv)
    bb = gp.bb
    src = bb._board[y1][x1]
    dst = bb._board[y2][x2]
    victim = _piece_value(dst)
    attacker = _piece_value(src)
    if victim <= 0:
        return 1.0
    # MVV/LVA：净得子力 + 对方损失相对我方投入
    gain = float(victim) - 0.28 * float(max(attacker, 1))
    # 映射到温和乘子区间，避免完全压死网络 prior
    t = math.tanh(gain / 900.0)
    return 1.0 + strength * (0.35 + 0.65 * max(0.0, t))


def tactical_boost_priors(
    gp: GamePlay,
    legals_s: list[str],
    priors: np.ndarray,
    *,
    check_mult: float = 3.4,
    capture_mult: float = 2.05,
    mvv_lva_strength: float = 0.55,
) -> np.ndarray:
    """
    对 ``priors`` 做乘性加权后归一化。

    - **将军**（含「将 + 吃」）：乘 ``check_mult``；若为吃子再乘 MVV/LVA 与 ``capture_mult``。
    - **非将吃子**：乘 ``capture_mult`` × MVV/LVA。
    - **其余**：保持网络 prior 相对比例。
    """
    p = np.asarray(priors, dtype=np.float64).copy()
    n = len(legals_s)
    if n == 0:
        return p
    p = np.maximum(p, 1e-16)
    for i, mv in enumerate(legals_s):
        m = 1.0
        chk = move_gives_check(gp, mv)
        cap = move_is_capture(gp, mv)
        if chk:
            m *= check_mult
        if cap:
            m *= capture_mult * _mvv_lva_capture_multiplier(gp, mv, strength=mvv_lva_strength)
        p[i] *= m
    s = float(np.sum(p))
    if s <= 0:
        p.fill(1.0 / n)
    else:
        p /= s
    return p
