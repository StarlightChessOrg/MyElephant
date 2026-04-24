"""
在棋子平面之外，补充与「棋规 / 地理 / 将帅安危 / 子力与灵活度」相关的输入通道，
让网络更容易接触「为何要走」的结构化线索（仍需训练目标配合，并非单靠特征即可学理）。

11 个理据平面（`encode_rationale_planes` 堆叠顺序），与 7 路有符号子力、``plane_extras`` 几何/步序/飞将/着法并集
一起在 ``encode_model_planes`` 中拼接；总通道 ``POLICY_SELECT_IN_CHANNELS`` = 7 + 11 + ``EXTRA_HINT_PLANE_COUNT``（见 ``plane_extras``）：

  1–3. 红九宫、黑九宫、黑方半场掩码；4. 行棋方 ±1；5–6. 帅位、将位；
  7. 被将军；8. 子力价值；9–11. 双方灵活度与行棋方着法质量。
"""
from __future__ import annotations

import numpy as np

from cchess.board import BaseChessBoard, ChessBoard
from cchess.piece import ChessSide, PieceT, fench_to_species

from my_elephant.chess.board_utils import chess_board_from_base
from my_elephant.chess.plane_extras import EXTRA_HINT_PLANE_COUNT

# 近似子力价值（车=9 量纲），将帅不参与数值以免干扰「净多子」感知
PIECE_VALUE_BY_FENCH: dict[str, float] = {
    "r": 9.0,
    "n": 4.0,
    "c": 4.5,
    "b": 2.0,
    "a": 2.0,
    "p": 1.0,
    "k": 0.0,
}

# 旧版 14 路己方/对方二值平面（encode_picker_planes）；分析脚本仍可参考
PIECE_PLANE_COUNT = 14
# encode_rationale_planes 的附加通道数（与 7 路子力、plane_extras 在 encode_model_planes 中拼接）
RATIONALE_PLANE_COUNT = 11
PIECE_SIGNED_PLANE_COUNT = 7

# 策略网络输入通道总数（见 features.encode_model_planes）
POLICY_SELECT_IN_CHANNELS = PIECE_SIGNED_PLANE_COUNT + RATIONALE_PLANE_COUNT + EXTRA_HINT_PLANE_COUNT
# 训练批填充：合法着法数上界（一般远小于此；若棋谱超出需调大）
POLICY_MAX_LEGAL_MOVES = 96
# 策略「先起点后落点」：ICCS 9×10 展平为 90 格（下标 y*9+x）
POLICY_GRID_NUMEL = 90
# 棋谱 ``RecordResult`` 解析：红方视角三分类（仅用于从 ``Head`` 读入后再转换）
RED_OUTCOME_WIN = 0
RED_OUTCOME_DRAW = 1
RED_OUTCOME_LOSS = 2
# 价值网络输出 / 训练标签：**当前行棋方**视角胜 / 和 / 负（交叉熵类下标）
STM_OUTCOME_WIN = 0
STM_OUTCOME_DRAW = 1
STM_OUTCOME_LOSS = 2
VALUE_LABEL_IGNORE = -100


def stm_outcome_class_from_red_outcome(red_cls: int, red_to_move: bool) -> int:
    """
    将棋谱终局的红方三分类 ``red_cls`` 转为**该步局面下行棋方**三分类。
    类下标仍为 ``0=胜,1=和,2=负``，语义相对**轮到走的一方**。
    """
    if red_cls == VALUE_LABEL_IGNORE:
        return VALUE_LABEL_IGNORE
    if red_to_move:
        return int(red_cls)
    if red_cls == RED_OUTCOME_WIN:
        return STM_OUTCOME_LOSS
    if red_cls == RED_OUTCOME_LOSS:
        return STM_OUTCOME_WIN
    return STM_OUTCOME_DRAW


def _fench_material_value(fench: str) -> float:
    ch = fench.lower()
    return float(PIECE_VALUE_BY_FENCH.get(ch, 0.0))


def _palace_red_mask() -> np.ndarray:
    """红方九宫（与 cchess 内部 y=0..2, x=3..5 一致），在 get_board_arr() 坐标下。"""
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in (0, 1, 2):
        for x in (3, 4, 5):
            m[9 - iy, x] = 1.0
    return m


def _palace_black_mask() -> np.ndarray:
    """黑方九宫（内部 y=7..9）。"""
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in (7, 8, 9):
        for x in (3, 4, 5):
            m[9 - iy, x] = 1.0
    return m


def _black_territory_mask() -> np.ndarray:
    """黑方半场（内部 y>=5），用于过河兵等地域概念。"""
    m = np.zeros((10, 9), dtype=np.float32)
    for iy in range(5, 10):
        m[9 - iy, :] = 1.0
    return m


def _king_planes(boardarr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """帅/将所在格为 1，其余 0（在 get_board_arr() 坐标下）。"""
    kr = np.zeros((10, 9), dtype=np.float32)
    kb = np.zeros((10, 9), dtype=np.float32)
    for r in range(10):
        for c in range(9):
            ch = boardarr[r, c]
            if ch == "K":
                kr[r, c] = 1.0
            elif ch == "k":
                kb[r, c] = 1.0
    return kr, kb


def _in_check_plane(cb: ChessBoard) -> np.ndarray:
    """当前行棋方是否被将军：整盘同值广播，便于卷积感知全局紧迫度。"""
    v = 1.0 if cb.is_checked() > 0 else 0.0
    return np.full((10, 9), v, dtype=np.float32)


def _side_to_move_plane(cb: ChessBoard) -> np.ndarray:
    """行棋方提示：红走全 +1，黑走全 −1（固定物理棋盘坐标）。"""
    if cb.move_side is None:
        return np.zeros((10, 9), dtype=np.float32)
    v = 1.0 if cb.move_side is not ChessSide.BLACK else -1.0
    return np.full((10, 9), v, dtype=np.float32)


def _signed_material_plane(cb: ChessBoard) -> np.ndarray:
    """行棋方视角：己方棋子格为 +v/9，对方为 −v/9；将帅格为 0。"""
    out = np.zeros((10, 9), dtype=np.float32)
    ms = cb.move_side
    if ms is None:
        return out
    denom = 9.0  # 按车归一，使数值大致落在 [-1, 1]
    for iy in range(10):
        for ix in range(9):
            fench = cb._board[iy][ix]
            if not fench:
                continue
            val = _fench_material_value(fench)
            if val <= 0.0:
                continue
            _, side = fench_to_species(fench)
            ar, ac = 9 - iy, ix
            sign = 1.0 if side == ms else -1.0
            out[ar, ac] = float(np.clip(sign * val / denom, -1.0, 1.0))
    return out


def _mobility_plane(cb: ChessBoard, side: ChessSide) -> np.ndarray:
    """
    指定方每个有子格子上的「合法着法数」/25 上限为 1。
    在 `cb` 的拷贝上临时设置 `move_side=side`，以符合 `is_valid_move_t` 的回合检查。
    """
    out = np.zeros((10, 9), dtype=np.float32)
    b = cb.copy()
    b.move_side = side
    for p in b.get_side_pieces(side):
        cnt = 0
        for mv in p.create_moves():
            if b.is_valid_move_t(mv):
                cnt += 1
        ar, ac = 9 - p.y, p.x
        out[ar, ac] = min(1.0, cnt / 25.0)
    return out


def _knight_target_score(tx: int, ty: int, side: ChessSide) -> float:
    """马走到的格子：边线、己方九宫心附近（易成窝心马）降分。"""
    s = 1.0
    if tx in (0, 8):
        s -= 0.45
    if side == ChessSide.RED and ty <= 2 and 3 <= tx <= 5:
        manh = abs(tx - 4) + abs(ty - 1)
        if manh <= 2:
            s -= 0.14 * (3 - manh)
    if side == ChessSide.BLACK and ty >= 7 and 3 <= tx <= 5:
        manh = abs(tx - 4) + abs(ty - 8)
        if manh <= 2:
            s -= 0.14 * (3 - manh)
    return float(np.clip(s, 0.08, 1.0))


def _knight_source_multiplier(px: int, py: int, side: ChessSide) -> float:
    """边马、压在己方九宫心一带的马（窝心马）整体降权。"""
    m = 1.0
    if px in (0, 8):
        m *= 0.84
    if side == ChessSide.RED and py <= 2 and 3 <= px <= 5:
        d = abs(px - 4) + abs(py - 1)
        if d <= 1:
            m *= 0.78
    if side == ChessSide.BLACK and py >= 7 and 3 <= px <= 5:
        d = abs(px - 4) + abs(py - 8)
        if d <= 1:
            m *= 0.78
    return float(np.clip(m, 0.55, 1.0))


def _move_quality(b: ChessBoard, side: ChessSide, piece, move_t: tuple) -> float:
    """单步合法着法的启发式质量 0~1（仅形状先验，非绝对棋理）。"""
    _, pos_to = move_t
    tx, ty = pos_to.x, pos_to.y
    sp = piece.species

    if sp == PieceT.KNIGHT:
        return _knight_target_score(tx, ty, side)

    if sp == PieceT.ROOK:
        s = 0.9 + 0.1 * (1.0 if tx == 4 else 0.0)
        return float(min(1.0, s))

    if sp == PieceT.CANNON:
        s = 0.9 + 0.06 * (1.0 if 2 <= tx <= 6 else 0.0)
        return float(min(1.0, s))

    if sp == PieceT.PAWN:
        forward = (ty > piece.y) if side == ChessSide.RED else (ty < piece.y)
        opp_half = ty >= 5 if side == ChessSide.RED else ty <= 4
        s = 0.86 + 0.07 * float(forward) + 0.09 * float(opp_half and forward)
        return float(min(1.0, s))

    if sp in (PieceT.BISHOP, PieceT.ADVISOR):
        return 0.93

    if sp == PieceT.KING:
        return 0.9

    return 0.92


def _mobility_quality_plane(cb: ChessBoard, side: ChessSide) -> np.ndarray:
    """
    行棋方（或指定方）每子一格：合法着法的平均「质量」分数，再经马位乘子压缩。
    与平面 9 的「着法数」互补：子多但着法「差」（如边马跳窝心）时两通道会分化。
    """
    out = np.zeros((10, 9), dtype=np.float32)
    b = cb.copy()
    b.move_side = side
    for p in b.get_side_pieces(side):
        quals: list[float] = []
        for mv in p.create_moves():
            if b.is_valid_move_t(mv):
                quals.append(_move_quality(b, side, p, mv))
        if not quals:
            continue
        avg = float(np.mean(quals))
        if p.species == PieceT.KNIGHT:
            avg *= _knight_source_multiplier(p.x, p.y, side)
        ar, ac = 9 - p.y, p.x
        out[ar, ac] = float(np.clip(avg, 0.0, 1.0))
    return out


def encode_rationale_planes(boardarr: np.ndarray, board_state: BaseChessBoard) -> np.ndarray:
    """
    返回 (RATIONALE_PLANE_COUNT, 10, 9) float32，与 boardarr / 棋子平面同一**固定红方**坐标系。
    """
    pr = _palace_red_mask()
    pb = _palace_black_mask()
    terr_b = _black_territory_mask()
    stm = np.zeros((10, 9), dtype=np.float32)
    kr, kb = _king_planes(boardarr)

    chk = np.zeros((10, 9), dtype=np.float32)
    mat = np.zeros((10, 9), dtype=np.float32)
    mob_self = np.zeros((10, 9), dtype=np.float32)
    mob_opp = np.zeros((10, 9), dtype=np.float32)
    mob_q_self = np.zeros((10, 9), dtype=np.float32)
    try:
        cb = chess_board_from_base(board_state)
        stm = _side_to_move_plane(cb)
        chk = _in_check_plane(cb)
        mat = _signed_material_plane(cb)
        if cb.move_side is not None:
            mob_self = _mobility_plane(cb, cb.move_side)
            mob_opp = _mobility_plane(cb, ChessSide.next_side(cb.move_side))
            mob_q_self = _mobility_quality_plane(cb, cb.move_side)
    except Exception:
        pass

    return np.stack(
        [pr, pb, terr_b, stm, kr, kb, chk, mat, mob_self, mob_opp, mob_q_self],
        axis=0,
    ).astype(np.float32)
