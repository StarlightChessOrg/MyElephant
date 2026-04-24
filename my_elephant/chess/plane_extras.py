"""策略网络额外输入平面：坐标归纳、局步进度、飞将、双方合法着法落点并集。"""

from __future__ import annotations

import numpy as np

from cchess.board import BaseChessBoard, ChessBoard
from cchess.piece import ChessSide

from my_elephant.chess.board_utils import chess_board_from_base

# 与 rationale 中常量相加得到 POLICY_SELECT_IN_CHANNELS（见 rationale.py）
EXTRA_HINT_PLANE_COUNT = 6


def _coord_planes() -> np.ndarray:
    """两路：x/8、y/9，与 ``get_board_arr()`` 矩阵下标一致。"""
    out = np.zeros((2, 10, 9), dtype=np.float32)
    for iy in range(10):
        for ix in range(9):
            out[0, iy, ix] = ix / 8.0
            out[1, iy, ix] = iy / 9.0
    return out


_COORD_CACHE: np.ndarray | None = None


def _coord_planes_cached() -> np.ndarray:
    global _COORD_CACHE
    if _COORD_CACHE is None:
        _COORD_CACHE = _coord_planes()
    return _COORD_CACHE


def _ply_broadcast_plane(move_index: int | None) -> np.ndarray:
    """本步在全局谱中的序号（从 0 起），归一化到 [0,1]；未知则为 0。"""
    if move_index is None or move_index < 0:
        v = 0.0
    else:
        v = min(1.0, float(move_index) / 150.0)
    return np.full((10, 9), v, dtype=np.float32)


def _kings_face_plane(boardarr: np.ndarray) -> np.ndarray:
    """
    飞将提示：帅与将同纵线且中间无子则全图 1，否则全 0。
    ``boardarr`` 与 ``BaseChessBoard.get_board_arr()`` 一致（y 轴已翻转）。
    """
    k_red: tuple[int, int] | None = None
    k_blk: tuple[int, int] | None = None
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if ch == "K":
                k_red = (ix, iy)
            elif ch == "k":
                k_blk = (ix, iy)
    if k_red is None or k_blk is None:
        return np.zeros((10, 9), dtype=np.float32)
    xr, yr = k_red
    xb, yb = k_blk
    if xr != xb:
        return np.zeros((10, 9), dtype=np.float32)
    y_lo, y_hi = (yr, yb) if yr < yb else (yb, yr)
    for y in range(y_lo + 1, y_hi):
        if boardarr[y, xr]:
            return np.zeros((10, 9), dtype=np.float32)
    return np.full((10, 9), 1.0, dtype=np.float32)


def _union_legal_move_destinations(cb: ChessBoard, side: ChessSide) -> np.ndarray:
    """该方任一合法着法的落点格并集（0/1），坐标与 ``get_board_arr()`` 一致。"""
    out = np.zeros((10, 9), dtype=np.float32)
    for p in cb.get_side_pieces(side):
        for mv in p.create_moves():
            if cb.is_valid_move_t(mv):
                _pf, pt = mv
                iy = 9 - pt.y
                ix = pt.x
                out[iy, ix] = 1.0
    return out


def encode_extra_hint_planes(
    boardarr: np.ndarray,
    board_state: BaseChessBoard,
    *,
    move_index: int | None = None,
) -> np.ndarray:
    """
    返回 ``(EXTRA_HINT_PLANE_COUNT, 10, 9)`` float32：

    0–1. 格坐标 x/8、y/9；2. 谱面步数归一化；3. 飞将；4–5. 行棋方 / 对方合法着法落点并集。
    """
    coord = _coord_planes_cached()
    ply = _ply_broadcast_plane(move_index)[np.newaxis, ...]
    face = _kings_face_plane(boardarr)[np.newaxis, ...]
    stm = np.zeros((1, 10, 9), dtype=np.float32)
    opp = np.zeros((1, 10, 9), dtype=np.float32)
    try:
        cb = chess_board_from_base(board_state)
        if cb.move_side is not None:
            stm[0] = _union_legal_move_destinations(cb, cb.move_side)
            opp[0] = _union_legal_move_destinations(cb, ChessSide.next_side(cb.move_side))
    except Exception:
        pass
    return np.concatenate([coord, ply, face, stm, opp], axis=0).astype(np.float32, copy=False)
