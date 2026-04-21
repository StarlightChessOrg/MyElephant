"""从 XML 棋谱生成策略网络训练样本。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import xmltodict

from cchess.board import BaseChessBoard
from cchess.move import Pos
from cchess.piece import ChessSide

from my_elephant.chess.features import encode_model_planes, parse_move_squares
from my_elephant.chess.rationale import POLICY_MAX_LEGAL_MOVES, POLICY_SELECT_IN_CHANNELS
from my_elephant.chess.session import legal_moves_iccs_for_board


def successor_planes_for_legals(
    bb: BaseChessBoard,
    legals: list[tuple[int, int, int, int]],
    feature_list: Mapping[str, list[str]],
) -> np.ndarray:
    """
    对 `bb` 当前局面下每个合法着法生成走后编码平面，顺序与 `legals` 一致。
    Returns (K, C, 10, 9) float32。

    使用 ``bb.copy()`` 而非 ``BaseChessBoard(bb.to_fen())``，避免每着一次 FEN 往返。
    """
    if not legals:
        return np.zeros((0, POLICY_SELECT_IN_CHANNELS, 10, 9), dtype=np.float32)
    rows: list[np.ndarray] = []
    for xa, ya, xb, yb in legals:
        trial = bb.copy()
        assert trial.move(Pos(xa, ya), Pos(xb, yb)) is not None
        trial.next_turn()
        red_after = trial.move_side is not ChessSide.BLACK
        boardarr = trial.get_board_arr()
        enc = encode_model_planes(boardarr, red_after, trial, feature_list)
        rows.append(enc.astype(np.float32, copy=False))
    return np.stack(rows, axis=0)


def _normalize_move_entries(move_node: Any) -> list[dict[str, Any]]:
    """xmltodict 在只有一个 Move 时返回 dict，多个时返回 list。"""
    if isinstance(move_node, list):
        return move_node
    if isinstance(move_node, dict):
        return [move_node]
    return []


def _load_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return xmltodict.parse(text)


def convert_game(
    onefile: str | Path,
    feature_list: Mapping[str, list[str]],
) -> Iterator[tuple[np.ndarray, int, int, bool]]:
    """
    逐步回放棋局。在棋谱着法执行前，枚举当前所有合法着法；对每个着法在**走完后的局面**
    上调用 `encode_model_planes` 得到平面，堆叠为 (K, C, 10, 9)。

    Yields:
        planes: (K, C, 10, 9) float32，K 为合法着法数（未填充）
        k_valid: K
        label_idx: 棋谱着法在 sorted(legals) 中的下标
        red_to_move: 走棋**之前**是否轮到红方（用于将 raw logit 解释为「对红方偏好」：红方 argmax、黑方 argmin）
    """
    doc = _load_record(onefile)
    fen = doc["ChineseChessRecord"]["Head"]["FEN"]
    moves_raw = doc["ChineseChessRecord"]["MoveList"]["Move"]
    moves = [
        m["@value"]
        for m in _normalize_move_entries(moves_raw)
        if m.get("@value") not in (None, "00-00")
    ]

    bb = BaseChessBoard(fen)
    for mv in moves:
        red_to_move = bb.move_side is not ChessSide.BLACK
        x1, y1, x2, y2 = parse_move_squares(mv)
        legals = sorted(legal_moves_iccs_for_board(bb))
        if len(legals) > POLICY_MAX_LEGAL_MOVES:
            raise ValueError(
                f"合法着法数 {len(legals)} 超过 POLICY_MAX_LEGAL_MOVES={POLICY_MAX_LEGAL_MOVES} file={onefile!r}"
            )
        played = (x1, y1, x2, y2)
        if played not in legals:
            raise ValueError(f"棋谱着法不在合法列表中: {mv!r} file={onefile!r}")
        label_idx = legals.index(played)

        planes = successor_planes_for_legals(bb, legals, feature_list)
        k_valid = int(planes.shape[0])

        moveresult = bb.move(Pos(x1, y1), Pos(x2, y2))
        assert moveresult is not None
        bb.next_turn()

        yield planes, k_valid, label_idx, red_to_move
