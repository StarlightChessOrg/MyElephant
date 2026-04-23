"""盘面到模型输入的平面特征编码（与旧版 notebook 逻辑一致）。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cchess.board import BaseChessBoard

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
    格上红方该兵种 **+1**，黑方 **-1**，空 **0**。策略头始终在同一坐标系下打分，无需「轮到谁」输入通道。
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
) -> np.ndarray:
    """
    策略网络输入：**仅 7 路有符号兵种平面**（红 +1 / 黑 -1），**固定红方视角**的盘面矩阵，不按行棋方翻转。
    走棋方信息不进入网络；策略为「先 ICCS 起点格、再落点格」两阶段分类（与对弈点击顺序一致）；价值头在训练标签侧为**行棋方**三分类（见 ``stm_outcome_class_from_red_outcome``）。

    ``red_to_move`` / ``board_state`` / ``feature_list`` 保留以兼容调用方，当前不参与本函数编码。
    """
    _ = (red_to_move, board_state, feature_list)
    return encode_signed_seven_planes(boardarr)
