"""从 XML 棋谱生成策略网络训练样本。

本仓库中的 ``.cbf`` 为 **XML** 文本（根元素 ``ChineseChessRecord``），而非象棋桥二进制 CBR/CBL。
``Head`` 中含 ``FEN``、``MoveList``；对局结果常见字段为 ``RecordResult``，编码与 ``cchess.reader_xqf`` 中
``result_dict`` 一致：``0`` 未知/未填，``1`` 红胜，``2`` 黑胜，``3``/``4`` 和棋。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import xmltodict

from cchess.board import BaseChessBoard
from cchess.move import Pos
from cchess.piece import ChessSide

from my_elephant.chess.features import encode_model_planes, parse_move_squares
from my_elephant.chess.rationale import (
    POLICY_MAX_LEGAL_MOVES,
    POLICY_SELECT_IN_CHANNELS,
    RED_OUTCOME_DRAW,
    RED_OUTCOME_LOSS,
    RED_OUTCOME_WIN,
    VALUE_LABEL_IGNORE,
)
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


def red_outcome_class_from_head(head: Any) -> int:
    """
    从 ``Head`` 节点解析红方对局结果三分类下标：胜 ``RED_OUTCOME_WIN``、和 ``RED_OUTCOME_DRAW``、负 ``RED_OUTCOME_LOSS``；
    未知或无法解析时返回 ``VALUE_LABEL_IGNORE``（不参与价值头交叉熵）。
    """
    if not isinstance(head, dict):
        return VALUE_LABEL_IGNORE
    rr = head.get("RecordResult")
    if rr is None:
        return VALUE_LABEL_IGNORE
    if isinstance(rr, dict):
        rr = rr.get("#text", rr.get("@value", ""))
    s = str(rr).strip()
    if not s:
        return VALUE_LABEL_IGNORE
    try:
        code = int(s)
    except ValueError:
        return VALUE_LABEL_IGNORE
    if code == 1:
        return RED_OUTCOME_WIN
    if code == 2:
        return RED_OUTCOME_LOSS
    if code in (3, 4):
        return RED_OUTCOME_DRAW
    return VALUE_LABEL_IGNORE


def convert_game(
    onefile: str | Path,
    feature_list: Mapping[str, list[str]],
) -> Iterator[tuple[np.ndarray, int, int, bool, np.ndarray, int]]:
    """
    逐步回放棋局。在棋谱着法执行前，枚举当前所有合法着法；对每个着法在**走完后的局面**
    上调用 `encode_model_planes` 得到平面，堆叠为 (K, C, 10, 9)。

    Yields:
        planes: (K, C, 10, 9) float32，K 为合法着法数（未填充）
        k_valid: K
        label_idx: 棋谱着法在 sorted(legals) 中的下标
        red_to_move: 走棋**之前**是否轮到红方（用于将 raw logit 解释为「对红方偏好」：红方 argmax、黑方 argmin）
        current_chw: (C, 10, 9) float32，走棋前当前局面编码
        outcome_cls: 红方胜/和/负类下标，或 ``VALUE_LABEL_IGNORE`` 表示无 ``RecordResult`` 标签
    """
    doc = _load_record(onefile)
    head = doc["ChineseChessRecord"]["Head"]
    fen = head["FEN"]
    outcome_cls = red_outcome_class_from_head(head)
    moves_raw = doc["ChineseChessRecord"]["MoveList"]["Move"]
    moves = [
        m["@value"]
        for m in _normalize_move_entries(moves_raw)
        if m.get("@value") not in (None, "00-00")
    ]

    bb = BaseChessBoard(fen)
    for mv in moves:
        red_to_move = bb.move_side is not ChessSide.BLACK
        boardarr_before = bb.get_board_arr()
        current_chw = encode_model_planes(
            boardarr_before, red_to_move, bb, feature_list
        ).astype(np.float32, copy=False)
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

        yield planes, k_valid, label_idx, red_to_move, current_chw, outcome_cls
