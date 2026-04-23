"""象棋盘面特征、棋谱迭代与对弈会话（围绕 `cchess` 规则引擎）。"""

from my_elephant.chess.features import (
    FEATURE_LIST,
    encode_model_planes,
    encode_picker_planes,
    encode_signed_seven_planes,
    orient_planes_for_model,
    parse_move_squares,
)
from my_elephant.chess.rationale import (
    PIECE_PLANE_COUNT,
    POLICY_GRID_NUMEL,
    POLICY_MAX_LEGAL_MOVES,
    POLICY_SELECT_IN_CHANNELS,
    RATIONALE_PLANE_COUNT,
    RED_OUTCOME_DRAW,
    RED_OUTCOME_LOSS,
    RED_OUTCOME_WIN,
    VALUE_LABEL_IGNORE,
)
from my_elephant.chess.session import GamePlay, legal_moves_iccs_for_board
from my_elephant.chess.xml_samples import (
    convert_game,
    iccs_flat_index,
    red_outcome_class_from_head,
    src_dst_masks_and_labels,
    successor_planes_for_legals,
)

__all__ = [
    "FEATURE_LIST",
    "PIECE_PLANE_COUNT",
    "RATIONALE_PLANE_COUNT",
    "POLICY_SELECT_IN_CHANNELS",
    "POLICY_MAX_LEGAL_MOVES",
    "POLICY_GRID_NUMEL",
    "RED_OUTCOME_WIN",
    "RED_OUTCOME_DRAW",
    "RED_OUTCOME_LOSS",
    "VALUE_LABEL_IGNORE",
    "encode_picker_planes",
    "encode_signed_seven_planes",
    "encode_model_planes",
    "orient_planes_for_model",
    "parse_move_squares",
    "convert_game",
    "iccs_flat_index",
    "src_dst_masks_and_labels",
    "red_outcome_class_from_head",
    "successor_planes_for_legals",
    "legal_moves_iccs_for_board",
    "GamePlay",
]
