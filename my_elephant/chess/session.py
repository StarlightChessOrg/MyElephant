"""人机对弈时维护棋盘并生成与训练一致的输入平面。"""

from __future__ import annotations

import numpy as np

from cchess.board import FULL_INIT_FEN, BaseChessBoard
from cchess.move import Pos
from cchess.piece import ChessSide

from my_elephant.chess.board_utils import chess_board_from_base
from my_elephant.chess.features import encode_model_planes


def legal_moves_iccs_for_board(bb: BaseChessBoard) -> list[tuple[int, int, int, int]]:
    """
    当前行棋方全部合法着法 (x1,y1,x2,y2)，ICCS 纵坐标，与 `parse_move_squares` / `BaseChessBoard.move` 一致。
    """
    cb = chess_board_from_base(bb)
    side = cb.move_side
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for piece in cb.get_side_pieces(side):
        for mv in piece.create_moves():
            pf, pt = mv
            if not cb.is_valid_move(pf, pt):
                continue
            if cb.is_checked_move(pf, pt):
                continue
            y1, y2 = 9 - pf.y, 9 - pt.y
            t = (pf.x, y1, pt.x, y2)
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


class GamePlay:
    def __init__(self) -> None:
        self.bb = BaseChessBoard(FULL_INIT_FEN)
        self.red = self.bb.move_side is not ChessSide.BLACK
        self._last_move_iccs: str | None = None

    def get_side(self) -> str:
        return "red" if self.red else "black"

    def legal_moves_iccs(self) -> list[tuple[int, int, int, int]]:
        return legal_moves_iccs_for_board(self.bb)

    def make_move(self, move: str) -> None:
        x1, y1, x2, y2 = int(move[0]), int(move[1]), int(move[3]), int(move[4])
        # 与 xml_samples.convert_game 一致：parse_move_squares 坐标直接交给 bb.move（含黑方）
        moveresult = self.bb.move(Pos(x1, y1), Pos(x2, y2))
        assert moveresult is not None
        self.bb.next_turn()
        self.red = not self.red
        self._last_move_iccs = move

    def print_board(self) -> None:
        self.bb.print_board()

    @property
    def last_move_iccs(self) -> str | None:
        """产生当前局面的上一手 ICCS；开局为 ``None``。"""
        return self._last_move_iccs

    def get_board_arr(self) -> np.ndarray:
        boardarr = self.bb.get_board_arr()
        # 与训练一致：固定红方物理视角 + 理据 + 扩展平面（含上一手）
        return encode_model_planes(
            boardarr, self.red, self.bb, last_move=self._last_move_iccs
        )
