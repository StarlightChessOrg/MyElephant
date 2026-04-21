"""棋盘结构拷贝，避免频繁的 to_fen / from_fen 解析。"""
from __future__ import annotations

import copy

from cchess.board import BaseChessBoard, ChessBoard


def chess_board_from_base(bb: BaseChessBoard) -> ChessBoard:
    """
    从内存中的 `BaseChessBoard`（含子类）构造 `ChessBoard`，供规则检查与棋理平面使用。
    比 ``ChessBoard(bb.to_fen())`` 少一次 FEN 序列化与词法解析。
    """
    cb = ChessBoard()
    cb._board = copy.deepcopy(bb._board)
    cb.move_side = bb.move_side
    cb.round = int(getattr(bb, "round", 1))
    return cb
