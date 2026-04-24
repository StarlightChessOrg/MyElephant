"""盘面到模型输入的平面特征编码（与旧版 notebook 逻辑一致）。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cchess.board import BaseChessBoard

from my_elephant.chess.plane_extras import encode_extra_hint_planes
from my_elephant.chess.rationale import encode_rationale_planes

# 与历史 notebook 中顺序保持一致
FEATURE_LIST: dict[str, list[str]] = {
    "red": ["A", "B", "C", "K", "N", "P", "R"],
    "black": ["a", "b", "c", "k", "n", "p", "r"],
}


def encode_picker_planes(
    boardarr: np.ndarray,
    red_to_move: bool,
    feature_list: Mapping[str, list[str]] | None = None,
) -> np.ndarray:
    """
    将棋盘字符矩阵编码为 (14, 10, 9) 的 uint8 平面：先己方 7 类，再对方 7 类。
    尚未做黑方视角的垂直翻转；翻转由调用方在「轮到黑走」时统一处理。
    """
    fl = FEATURE_LIST if feature_list is None else feature_list
    planes: list[np.ndarray] = []
    if red_to_move:
        for ch in fl["red"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
        for ch in fl["black"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
    else:
        for ch in fl["black"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
        for ch in fl["red"]:
            planes.append(np.asarray(boardarr == ch, dtype=np.uint8))
    return np.asarray(planes, dtype=np.uint8)


def orient_planes_for_model(planes: np.ndarray, red_to_move: bool) -> np.ndarray:
    """黑方走棋时沿 y 轴翻转，使网络始终面对「自下而上」的己方半场。"""
    if red_to_move:
        return planes
    return planes[:, ::-1, :]


def parse_move_squares(move: str) -> tuple[int, int, int, int]:
    """解析如 '77-47' 的坐标串为 (x1, y1, x2, y2)。"""
    if len(move) < 5 or move[2] != "-":
        raise ValueError(f"无法解析走法: {move!r}")
    x1, y1, x2, y2 = int(move[0]), int(move[1]), int(move[3]), int(move[4])
    return x1, y1, x2, y2


def encode_signed_seven_planes(boardarr: np.ndarray) -> np.ndarray:
    """
    七种兵种各一路，**固定红方/物理棋盘视角**（与 ``get_board_arr()`` 矩阵一致），不因轮到谁走而翻转。
    通道顺序：仕 A、相 B、炮 C、帅 K、马 N、兵 P、车 R；
    格上红方该兵种 **+1**，黑方 **-1**，空 **0**。策略头在同一坐标系下打分；行棋方等全局线索由 ``encode_model_planes`` 拼接的理据平面提供。
    """
    pairs = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("K", "k"),
        ("N", "n"),
        ("P", "p"),
        ("R", "r"),
    ]
    out = np.zeros((7, 10, 9), dtype=np.float32)
    for i, (ru, bk) in enumerate(pairs):
        out[i] = (boardarr == ru).astype(np.float32) - (boardarr == bk).astype(np.float32)
    return out


def encode_model_planes(
    boardarr: np.ndarray,
    red_to_move: bool,
    board_state: BaseChessBoard,
    feature_list: Mapping[str, list[str]] | None = None,
    *,
    move_index: int | None = None,
    last_move: str | None = None,
) -> np.ndarray:
    """
    策略网络输入（**固定红方物理棋盘**坐标，不按行棋方翻转棋盘）：

    - **7 路** 有符号兵种；**11 路** 理据；**``EXTRA_HINT_PLANE_COUNT`` 路** ``plane_extras``
      （坐标/步序/飞将/着法并集/上一手/子力与将几何/吃子与兵种控制/河界与半场/将邻与象士/兵吃与将射线/双方各兵种数量广播等，见 ``plane_extras.encode_extra_hint_planes`` 文档串）。

    总通道 ``POLICY_SELECT_IN_CHANNELS`` = 7 + 11 + ``EXTRA_HINT_PLANE_COUNT``。``last_move`` 为产生当前局面的上一手 ICCS 串（如 ``77-67``），无则 ``None``。
    """
    _ = (red_to_move, feature_list)
    pieces = encode_signed_seven_planes(boardarr)
    rationale = encode_rationale_planes(boardarr, board_state)
    extra = encode_extra_hint_planes(
        boardarr, board_state, move_index=move_index, last_move=last_move
    )
    return np.concatenate([pieces, rationale, extra], axis=0)
