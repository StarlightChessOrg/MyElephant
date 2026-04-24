"""策略网络额外输入平面：坐标、步序、飞将、着法并集、上一手、子力、将距、吃子目标、车马炮控制等。"""

from __future__ import annotations

import numpy as np

from cchess.board import BaseChessBoard, ChessBoard
from cchess.piece import ChessSide, PieceT, fench_to_species

from my_elephant.chess.board_utils import chess_board_from_base

# 与 rationale 中常量相加得到 POLICY_SELECT_IN_CHANNELS（见 rationale.py）
EXTRA_HINT_PLANE_COUNT = 47

# 与 rationale.PIECE_VALUE_BY_FENCH 量纲一致（将帅 0），避免 rationale↔plane_extras 循环 import
_MAT_SUM_DENOM = 55.0  # 约两车+子力上界，用于归一化总子力广播


def _mat_val(ch: str) -> float:
    if not ch:
        return 0.0
    key = ch.lower()
    m = {"r": 9.0, "n": 4.0, "c": 4.5, "b": 2.0, "a": 2.0, "p": 1.0, "k": 0.0}
    return float(m.get(key, 0.0))


def _coord_planes() -> np.ndarray:
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
    if move_index is None or move_index < 0:
        v = 0.0
    else:
        v = min(1.0, float(move_index) / 150.0)
    return np.full((10, 9), v, dtype=np.float32)


def _kings_face_plane(boardarr: np.ndarray) -> np.ndarray:
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


def _cb_for_side(cb0: ChessBoard, side: ChessSide) -> ChessBoard:
    cb = cb0.copy()
    cb.move_side = side
    return cb


def _union_legal_move_destinations(cb0: ChessBoard, side: ChessSide) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    cb = _cb_for_side(cb0, side)
    for p in cb.get_side_pieces(side):
        for mv in p.create_moves():
            if cb.is_valid_move_t(mv):
                _pf, pt = mv
                iy = 9 - pt.y
                ix = pt.x
                out[iy, ix] = 1.0
    return out


def _last_move_planes(last_move: str | None) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((10, 9), dtype=np.float32)
    b = np.zeros((10, 9), dtype=np.float32)
    if not last_move or len(last_move) < 5 or last_move[2] != "-":
        return a, b
    from my_elephant.chess.features import parse_move_squares

    try:
        x1, y1, x2, y2 = parse_move_squares(last_move)
    except ValueError:
        return a, b
    a[y1, x1] = 1.0
    b[y2, x2] = 1.0
    return a, b


def _material_broadcast(boardarr: np.ndarray, stm: ChessSide) -> tuple[np.ndarray, np.ndarray]:
    s_stm = 0.0
    s_opp = 0.0
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if not ch:
                continue
            v = _mat_val(ch)
            if v <= 0.0:
                continue
            _sp, side = fench_to_species(ch)
            if side == stm:
                s_stm += v
            else:
                s_opp += v
    v_stm = min(1.0, s_stm / _MAT_SUM_DENOM)
    v_opp = min(1.0, s_opp / _MAT_SUM_DENOM)
    return np.full((10, 9), v_stm, dtype=np.float32), np.full((10, 9), v_opp, dtype=np.float32)


def _king_geometry(boardarr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将帅曼哈顿距离 /17 广播；是否同横排（iy 相同）广播 0/1。"""
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
        z = np.zeros((10, 9), dtype=np.float32)
        return z, z
    xr, yr = k_red
    xb, yb = k_blk
    dist = abs(xr - xb) + abs(yr - yb)
    dn = min(1.0, float(dist) / 17.0)
    same_rank = 1.0 if yr == yb else 0.0
    return np.full((10, 9), dn, dtype=np.float32), np.full((10, 9), same_rank, dtype=np.float32)


def _board_density(boardarr: np.ndarray) -> np.ndarray:
    n = sum(1 for iy in range(10) for ix in range(9) if boardarr[iy, ix])
    return np.full((10, 9), float(n) / 90.0, dtype=np.float32)


def _major_ratio(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    majors = (PieceT.ROOK, PieceT.KNIGHT, PieceT.CANNON)
    c = 0
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if not ch:
                continue
            sp, sd = fench_to_species(ch)
            if sd == side and sp in majors:
                c += 1
    return np.full((10, 9), min(1.0, c / 6.0), dtype=np.float32)


def _capture_destination_union(cb0: ChessBoard, attacker: ChessSide) -> np.ndarray:
    """攻击方所有「吃子」合法着法的落点并集（0/1）。"""
    out = np.zeros((10, 9), dtype=np.float32)
    cb = _cb_for_side(cb0, attacker)
    opp = ChessSide.next_side(attacker)
    for p in cb.get_side_pieces(attacker):
        for mv in p.create_moves():
            if not cb.is_valid_move_t(mv):
                continue
            pf, pt = mv
            f_to = cb._board[pt.y][pt.x]
            if not f_to:
                continue
            _sp_to, side_to = fench_to_species(f_to)
            if side_to != opp:
                continue
            iy = 9 - pt.y
            ix = pt.x
            out[iy, ix] = 1.0
    return out


def _pawn_progress_plane(boardarr: np.ndarray) -> np.ndarray:
    """兵/卒：红兵 ``(9-iy)/9``，黑卒 ``iy/9``，其余格 0。"""
    out = np.zeros((10, 9), dtype=np.float32)
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if ch == "P":
                out[iy, ix] = (9.0 - float(iy)) / 9.0
            elif ch == "p":
                out[iy, ix] = float(iy) / 9.0
    return out


def _piece_attack_union(cb0: ChessBoard, side: ChessSide, species: PieceT) -> np.ndarray:
    """指定方某兵种能走到的所有合法落点（含吃与不吃），需在 ``cb`` 上临时 ``move_side=side``。"""
    out = np.zeros((10, 9), dtype=np.float32)
    cb = _cb_for_side(cb0, side)
    for p in cb.get_side_pieces(side):
        if p.species != species:
            continue
        for mv in p.create_moves():
            if cb.is_valid_move_t(mv):
                _pf, pt = mv
                iy = 9 - pt.y
                ix = pt.x
                out[iy, ix] = 1.0
    return out


def _river_band_plane() -> np.ndarray:
    """楚河汉界附近两横排（``iy`` 4、5）标 1。"""
    out = np.zeros((10, 9), dtype=np.float32)
    out[4, :] = 1.0
    out[5, :] = 1.0
    return out


def _half_board_planes() -> tuple[np.ndarray, np.ndarray]:
    """红方半场（``iy>=5``）、黑方半场（``iy<=4``），与 ``get_board_arr`` 纵轴一致。"""
    red = np.zeros((10, 9), dtype=np.float32)
    blk = np.zeros((10, 9), dtype=np.float32)
    red[5:, :] = 1.0
    blk[:5, :] = 1.0
    return red, blk


def _find_king_iccs(boardarr: np.ndarray, side: ChessSide) -> tuple[int, int] | None:
    target = "K" if side == ChessSide.RED else "k"
    for iy in range(10):
        for ix in range(9):
            if boardarr[iy, ix] == target:
                return ix, iy
    return None


def _king_ortho_neighbor_density(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    """帅/将四正邻有子数 / 4，全图广播。"""
    pos = _find_king_iccs(boardarr, side)
    if pos is None:
        return np.zeros((10, 9), dtype=np.float32)
    kx, ky = pos
    n = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = kx + dx, ky + dy
        if 0 <= nx < 9 and 0 <= ny < 10 and boardarr[ny, nx]:
            n += 1
    v = n / 4.0
    return np.full((10, 9), v, dtype=np.float32)


def _king_cross_empty_rays(boardarr: np.ndarray, side: ChessSide) -> np.ndarray:
    """从己方帅/将出发四向射线，直到挡子为止，中间空格标 1。"""
    out = np.zeros((10, 9), dtype=np.float32)
    pos = _find_king_iccs(boardarr, side)
    if pos is None:
        return out
    kx, ky = pos
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        s = 1
        while True:
            nx, ny = kx + dx * s, ky + dy * s
            if not (0 <= nx < 9 and 0 <= ny < 10):
                break
            if boardarr[ny, nx]:
                break
            out[ny, nx] = 1.0
            s += 1
    return out


def _pawn_capture_only_union(cb0: ChessBoard, side: ChessSide) -> np.ndarray:
    """兵/卒仅吃子的合法落点并集。"""
    out = np.zeros((10, 9), dtype=np.float32)
    cb = _cb_for_side(cb0, side)
    opp = ChessSide.next_side(side)
    for p in cb.get_side_pieces(side):
        if p.species != PieceT.PAWN:
            continue
        for mv in p.create_moves():
            if not cb.is_valid_move_t(mv):
                continue
            pf, pt = mv
            f_to = cb._board[pt.y][pt.x]
            if not f_to:
                continue
            _st, side_to = fench_to_species(f_to)
            if side_to != opp:
                continue
            iy = 9 - pt.y
            ix = pt.x
            out[iy, ix] = 1.0
    return out


def _bishop_advisor_union(cb0: ChessBoard, side: ChessSide) -> np.ndarray:
    out = np.zeros((10, 9), dtype=np.float32)
    cb = _cb_for_side(cb0, side)
    for sp in (PieceT.BISHOP, PieceT.ADVISOR):
        for p in cb.get_side_pieces(side):
            if p.species != sp:
                continue
            for mv in p.create_moves():
                if cb.is_valid_move_t(mv):
                    _pf, pt = mv
                    iy = 9 - pt.y
                    ix = pt.x
                    out[iy, ix] = 1.0
    return out


def _count_species(boardarr: np.ndarray, side: ChessSide, species: PieceT) -> int:
    c = 0
    for iy in range(10):
        for ix in range(9):
            ch = boardarr[iy, ix]
            if not ch:
                continue
            sp, sd = fench_to_species(ch)
            if sd == side and sp == species:
                c += 1
    return c


def _species_count_broadcast(
    boardarr: np.ndarray, side: ChessSide, species: PieceT, denom: float
) -> np.ndarray:
    c = _count_species(boardarr, side, species)
    return np.full((10, 9), min(1.0, float(c) / denom), dtype=np.float32)


def encode_extra_hint_planes(
    boardarr: np.ndarray,
    board_state: BaseChessBoard,
    *,
    move_index: int | None = None,
    last_move: str | None = None,
) -> np.ndarray:
    """
    返回 ``(EXTRA_HINT_PLANE_COUNT, 10, 9)`` float32。

    0–23：坐标、步序、飞将、着法并集、上一手、子力与将几何、占子、大子比、吃子目标、兵纵深、双方车马炮控制。

    24–46：河界带；红/黑半场；双方将四邻有子密度；兵卒吃子落点；象士合法落点并集；将四向空射线；
    双方车/马/炮/象/士/兵个数广播（各除以典型上界 2,2,2,2,2,5）。
    """
    coord = _coord_planes_cached()
    ply = _ply_broadcast_plane(move_index)[np.newaxis, ...]
    face = _kings_face_plane(boardarr)[np.newaxis, ...]
    stm_d = np.zeros((1, 10, 9), dtype=np.float32)
    opp_d = np.zeros((1, 10, 9), dtype=np.float32)
    lf = np.zeros((1, 10, 9), dtype=np.float32)
    lt = np.zeros((1, 10, 9), dtype=np.float32)
    mstm = np.zeros((1, 10, 9), dtype=np.float32)
    mopp = np.zeros((1, 10, 9), dtype=np.float32)
    kdist = np.zeros((1, 10, 9), dtype=np.float32)
    krank = np.zeros((1, 10, 9), dtype=np.float32)
    dens = np.zeros((1, 10, 9), dtype=np.float32)
    maj_s = np.zeros((1, 10, 9), dtype=np.float32)
    maj_o = np.zeros((1, 10, 9), dtype=np.float32)
    cap_s = np.zeros((1, 10, 9), dtype=np.float32)
    cap_o = np.zeros((1, 10, 9), dtype=np.float32)
    pwn = np.zeros((1, 10, 9), dtype=np.float32)
    orook = np.zeros((1, 10, 9), dtype=np.float32)
    ocann = np.zeros((1, 10, 9), dtype=np.float32)
    oknight = np.zeros((1, 10, 9), dtype=np.float32)
    srook = np.zeros((1, 10, 9), dtype=np.float32)
    scann = np.zeros((1, 10, 9), dtype=np.float32)
    sknight = np.zeros((1, 10, 9), dtype=np.float32)

    river = _river_band_plane()[np.newaxis, ...]
    red_h, blk_h = _half_board_planes()
    rh = red_h[np.newaxis, ...]
    bh = blk_h[np.newaxis, ...]
    knei_s = np.zeros((1, 10, 9), dtype=np.float32)
    knei_o = np.zeros((1, 10, 9), dtype=np.float32)
    pcap_s = np.zeros((1, 10, 9), dtype=np.float32)
    pcap_o = np.zeros((1, 10, 9), dtype=np.float32)
    ba_s = np.zeros((1, 10, 9), dtype=np.float32)
    ba_o = np.zeros((1, 10, 9), dtype=np.float32)
    kray_s = np.zeros((1, 10, 9), dtype=np.float32)
    kray_o = np.zeros((1, 10, 9), dtype=np.float32)
    z1 = np.zeros((1, 10, 9), dtype=np.float32)
    stm_counts: list[np.ndarray] = [z1] * 6
    opp_counts: list[np.ndarray] = [z1] * 6

    lfa, ltb = _last_move_planes(last_move)
    lf[0] = lfa
    lt[0] = ltb

    dens[0] = _board_density(boardarr)
    pwn[0] = _pawn_progress_plane(boardarr)
    kd, kr = _king_geometry(boardarr)
    kdist[0] = kd
    krank[0] = kr

    try:
        cb = chess_board_from_base(board_state)
        if cb.move_side is not None:
            stm_side = cb.move_side
            opp_side = ChessSide.next_side(stm_side)
            stm_d[0] = _union_legal_move_destinations(cb, stm_side)
            opp_d[0] = _union_legal_move_destinations(cb, opp_side)
            ms, mo = _material_broadcast(boardarr, stm_side)
            mstm[0] = ms
            mopp[0] = mo
            maj_s[0] = _major_ratio(boardarr, stm_side)
            maj_o[0] = _major_ratio(boardarr, opp_side)
            cap_s[0] = _capture_destination_union(cb, stm_side)
            cap_o[0] = _capture_destination_union(cb, opp_side)
            orook[0] = _piece_attack_union(cb, opp_side, PieceT.ROOK)
            ocann[0] = _piece_attack_union(cb, opp_side, PieceT.CANNON)
            oknight[0] = _piece_attack_union(cb, opp_side, PieceT.KNIGHT)
            srook[0] = _piece_attack_union(cb, stm_side, PieceT.ROOK)
            scann[0] = _piece_attack_union(cb, stm_side, PieceT.CANNON)
            sknight[0] = _piece_attack_union(cb, stm_side, PieceT.KNIGHT)

            knei_s[0] = _king_ortho_neighbor_density(boardarr, stm_side)
            knei_o[0] = _king_ortho_neighbor_density(boardarr, opp_side)
            pcap_s[0] = _pawn_capture_only_union(cb, stm_side)
            pcap_o[0] = _pawn_capture_only_union(cb, opp_side)
            ba_s[0] = _bishop_advisor_union(cb, stm_side)
            ba_o[0] = _bishop_advisor_union(cb, opp_side)
            kray_s[0] = _king_cross_empty_rays(boardarr, stm_side)
            kray_o[0] = _king_cross_empty_rays(boardarr, opp_side)

            den = (2.0, 2.0, 2.0, 2.0, 2.0, 5.0)
            specs = (
                PieceT.ROOK,
                PieceT.KNIGHT,
                PieceT.CANNON,
                PieceT.BISHOP,
                PieceT.ADVISOR,
                PieceT.PAWN,
            )
            stm_counts = [
                _species_count_broadcast(boardarr, stm_side, sp, d)[np.newaxis, ...]
                for sp, d in zip(specs, den)
            ]
            opp_counts = [
                _species_count_broadcast(boardarr, opp_side, sp, d)[np.newaxis, ...]
                for sp, d in zip(specs, den)
            ]
    except Exception:
        pass

    return np.concatenate(
        [
            coord,
            ply,
            face,
            stm_d,
            opp_d,
            lf,
            lt,
            mstm,
            mopp,
            kdist,
            krank,
            dens,
            maj_s,
            maj_o,
            cap_s,
            cap_o,
            pwn,
            orook,
            ocann,
            oknight,
            srook,
            scann,
            sknight,
            river,
            rh,
            bh,
            knei_s,
            knei_o,
            pcap_s,
            pcap_o,
            ba_s,
            ba_o,
            kray_s,
            kray_o,
            *stm_counts,
            *opp_counts,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
