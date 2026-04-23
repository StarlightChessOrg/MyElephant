"""MCTS 自对弈单局：生成与 ``convert_game`` 一致的训练元组（终局红方三分类标签在局末回填）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cchess.piece import ChessSide

from my_elephant.chess.board_utils import chess_board_from_base
from my_elephant.chess.features import encode_model_planes, parse_move_squares
from my_elephant.chess.rationale import RED_OUTCOME_DRAW, RED_OUTCOME_LOSS, RED_OUTCOME_WIN
from my_elephant.chess.session import GamePlay, legal_moves_iccs_for_board
from my_elephant.chess.xml_samples import src_dst_masks_and_labels
from my_elephant.training.mcts_engine import copy_gameplay, mcts_search
from my_elephant.training.policy_torch import eval_policy_value_at_root

if TYPE_CHECKING:
    from my_elephant.training.policy_torch import SuccessorPolicy


def terminal_red_outcome_class(gp: GamePlay) -> int | None:
    """若已终局返回红方结果类；否则 ``None``。"""
    if gp.legal_moves_iccs():
        return None
    cb = chess_board_from_base(gp.bb)
    if cb.is_checkmate():
        if gp.get_side() == "red":
            return RED_OUTCOME_LOSS
        return RED_OUTCOME_WIN
    return RED_OUTCOME_DRAW


def build_sample_before_move(
    gp: GamePlay,
    move_iccs: str,
    flist: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.int64, bool]:
    """当前 ``gp`` 未走时，记录走 ``move_iccs`` 的监督样本（不含终局类）。"""
    bb = gp.bb
    red_to_move = bb.move_side is not ChessSide.BLACK
    boardarr_before = bb.get_board_arr()
    current_chw = encode_model_planes(boardarr_before, red_to_move, bb, flist).astype(np.float32, copy=False)
    legals = sorted(legal_moves_iccs_for_board(bb))
    x1, y1, x2, y2 = parse_move_squares(move_iccs)
    played = (x1, y1, x2, y2)
    if played not in legals:
        raise RuntimeError(f"自对弈着法不在合法集: {move_iccs!r}")
    src_mask, dst_mask, li_s, li_d = src_dst_masks_and_labels(legals, played)
    cur_hwc = np.transpose(current_chw, (1, 2, 0)).astype(np.float32, copy=False)
    return (
        cur_hwc,
        src_mask,
        dst_mask,
        np.int64(li_s),
        np.int64(li_d),
        red_to_move,
    )


def play_one_selfplay_mcts_game(
    model: "SuccessorPolicy",
    device,
    flist: dict[str, list[str]],
    *,
    mcts_sims: int = 128,
    c_puct: float = 1.5,
    mcts_max_seconds: float | None = 8.0,
    mcts_workers: int | None = 1,
    thread_pool=None,
    max_plies: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.int64, bool, int]]:
    """
    双方同模型 MCTS 自对弈一局；每步用 MCTS 最优着作为监督标签；终局后回填红方结果类。
    返回与 ``collate_twohead_policy_value_batch`` 兼容的样本列表。
    """
    game = GamePlay()
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.int64, np.int64, bool]] = []

    def ev(gp: GamePlay):
        return eval_policy_value_at_root(gp, model, device, flist)

    for _ in range(max_plies):
        oc = terminal_red_outcome_class(game)
        if oc is not None:
            return [(*s, np.int64(oc)) for s in pending]
        legals = game.legal_moves_iccs()
        if not legals:
            oc = terminal_red_outcome_class(game)
            assert oc is not None
            return [(*s, np.int64(oc)) for s in pending]

        mv, _st, _root = mcts_search(
            copy_gameplay(game),
            ev,
            n_simulations=mcts_sims,
            c_puct=c_puct,
            max_seconds=mcts_max_seconds,
            virtual_loss=3.0,
            n_workers=mcts_workers,
            thread_pool=thread_pool,
            reuse_subtree=None,
        )
        pending.append(build_sample_before_move(game, mv, flist))
        game.make_move(mv)

    oc = RED_OUTCOME_DRAW
    return [(*s, np.int64(oc)) for s in pending]
