"""
Tkinter 图形对弈：圆形棋子、鼠标走子、对手上一手箭头提示；
红/黑可分别选「人类」「纯网络」「MCTS+双头网络」。
"""
from __future__ import annotations

import argparse
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import numpy as np
import torch

from my_elephant.chess import FEATURE_LIST, GamePlay
from my_elephant.chess.features import parse_move_squares
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS
from my_elephant.chess.xml_samples import successor_planes_for_legals
from my_elephant.training.mcts_engine import copy_gameplay, mcts_search
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    batched_successors_nhwc_to_torch,
    eval_policy_value_at_root,
    logits_as_red_preference,
    torch_load_checkpoint,
)

STRATEGY_HUMAN = "人类"
STRATEGY_NEURAL = "纯网络"
STRATEGY_MCTS = "MCTS+双头网络"
STRATEGIES = (STRATEGY_HUMAN, STRATEGY_NEURAL, STRATEGY_MCTS)

# 棋子显示（红大写 / 黑小写 → 同一汉字，靠颜色区分）
_PIECE_CHAR = {
    "R": "车",
    "N": "马",
    "B": "相",
    "A": "仕",
    "K": "帅",
    "C": "炮",
    "P": "兵",
    "r": "车",
    "n": "马",
    "b": "象",
    "a": "士",
    "k": "将",
    "c": "炮",
    "p": "卒",
}


def _infer_filters_from_state(sd: dict) -> int:
    w = sd.get("stem_conv.weight")
    if w is not None:
        return int(w.shape[0])
    return 64


def _piece_side(ch: str | None) -> str | None:
    if not ch:
        return None
    return "red" if ch.isupper() else "black"


def _neural_pick_move(gp: GamePlay, model: SuccessorPolicy, device: torch.device, flist: dict) -> str:
    legals_t = sorted(gp.legal_moves_iccs())
    if not legals_t:
        raise RuntimeError("无合法着法")
    legals_s = [f"{a}{b}-{c}{d}" for (a, b, c, d) in legals_t]
    planes_k = successor_planes_for_legals(gp.bb, legals_t, flist)
    x_hwc = np.transpose(planes_k, (0, 2, 3, 1))
    x = np.expand_dims(x_hwc, axis=0)
    xt = batched_successors_nhwc_to_torch(x, device)
    logits = model.predict_move_logits(xt)
    red = gp.get_side() == "red"
    logits_r = logits_as_red_preference(logits, red)
    li = int(torch.argmax(logits_r).item())
    return legals_s[li]


class XiangqiTkApp:
    def __init__(
        self,
        master: tk.Tk,
        model: SuccessorPolicy,
        device: torch.device,
        flist: dict[str, list[str]],
        mcts_sims: int,
        c_puct: float,
    ) -> None:
        self.master = master
        self.model = model
        self.device = device
        self.flist = flist
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.game = GamePlay()
        self.sel_from: tuple[int, int] | None = None
        self.last_move: tuple[int, int, int, int] | None = None
        self._ai_busy = False

        master.title("MyElephant 象棋对弈")
        self.CELL = 52
        self.OFF = 36
        self.CW = self.OFF * 2 + self.CELL * 8 + 1
        self.CH = self.OFF * 2 + self.CELL * 9 + 1

        main = ttk.Frame(master, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(left, width=self.CW, height=self.CH, bg="#e8d4b8", highlightthickness=1)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)

        right = ttk.Frame(main, padding=(12, 0))
        right.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(right, text="红方策略").pack(anchor=tk.W)
        self.var_red = tk.StringVar(value=STRATEGY_HUMAN)
        cb_r = ttk.Combobox(right, textvariable=self.var_red, values=STRATEGIES, state="readonly", width=14)
        cb_r.pack(anchor=tk.W, pady=(0, 8))
        cb_r.bind("<<ComboboxSelected>>", lambda _e: master.after(50, self._maybe_schedule_ai))

        ttk.Label(right, text="黑方策略").pack(anchor=tk.W)
        self.var_black = tk.StringVar(value=STRATEGY_NEURAL)
        cb_b = ttk.Combobox(right, textvariable=self.var_black, values=STRATEGIES, state="readonly", width=14)
        cb_b.pack(anchor=tk.W, pady=(0, 8))
        cb_b.bind("<<ComboboxSelected>>", lambda _e: master.after(50, self._maybe_schedule_ai))

        ttk.Button(right, text="新局", command=self._new_game).pack(fill=tk.X, pady=4)
        self.status = ttk.Label(right, text="", wraplength=200)
        self.status.pack(anchor=tk.W, pady=8)

        self._draw_static_grid()
        self._refresh_board()

    def _iccs_center(self, ix: int, iy: int) -> tuple[float, float]:
        h = self.CELL / 2.0
        cx = self.OFF + ix * self.CELL + h
        cy = self.OFF + iy * self.CELL + h
        return cx, cy

    def _draw_static_grid(self) -> None:
        c = self.canvas
        for i in range(10):
            y = self.OFF + i * self.CELL
            c.create_line(self.OFF, y, self.OFF + 8 * self.CELL, y, fill="#5c4033", width=1)
        for i in range(9):
            x = self.OFF + i * self.CELL
            y0, y1 = self.OFF, self.OFF + 9 * self.CELL
            c.create_line(x, y0, x, y1, fill="#5c4033", width=1)
        # 河界
        rx0, ry = self.OFF, self.OFF + 4 * self.CELL + self.CELL // 2
        c.create_text(rx0 + 2 * self.CELL, ry, text="楚 河", fill="#6b5344", font=("Microsoft YaHei", 11))
        c.create_text(rx0 + 6 * self.CELL, ry, text="漢 界", fill="#6b5344", font=("Microsoft YaHei", 11))

    def _raw_board(self) -> np.ndarray:
        return np.asarray(self.game.bb._board[::-1])

    def _refresh_board(self) -> None:
        self.canvas.delete("piece")
        self.canvas.delete("arrow")
        self.canvas.delete("selhi")
        arr = self._raw_board()
        r = self.CELL // 2 - 4
        for iy in range(10):
            for ix in range(9):
                ch = arr[iy, ix]
                if not ch:
                    continue
                cx, cy = self._iccs_center(ix, iy)
                side = _piece_side(str(ch))
                fill = "#c62828" if side == "red" else "#1565c0"
                outline = "#3e2723"
                self.canvas.create_oval(
                    cx - r,
                    cy - r,
                    cx + r,
                    cy + r,
                    fill=fill,
                    outline=outline,
                    width=2,
                    tags="piece",
                )
                label = _PIECE_CHAR.get(str(ch), "?")
                self.canvas.create_text(
                    cx, cy, text=label, fill="white", font=("Microsoft YaHei", 14, "bold"), tags="piece"
                )

        if self.last_move is not None:
            x1, y1, x2, y2 = self.last_move
            ax, ay = self._iccs_center(x1, y1)
            bx, by = self._iccs_center(x2, y2)
            self.canvas.create_line(ax, ay, bx, by, fill="#ff9800", width=3, arrow=tk.LAST, tags="arrow")

        if self.sel_from is not None:
            sx, sy = self.sel_from
            cx, cy = self._iccs_center(sx, sy)
            d = self.CELL // 2 - 2
            self.canvas.create_rectangle(
                cx - d, cy - d, cx + d, cy + d, outline="#ffeb3b", width=3, tags="selhi"
            )

        side = "红方" if self.game.get_side() == "red" else "黑方"
        self.status.config(text=f"轮到 {side} 走棋")

    def _pixel_to_iccs(self, px: float, py: float) -> tuple[int, int] | None:
        ix = int((px - self.OFF) // self.CELL)
        iy = int((py - self.OFF) // self.CELL)
        if 0 <= ix <= 8 and 0 <= iy <= 9:
            return ix, iy
        return None

    def _strategy_for_current(self) -> str:
        return self.var_red.get() if self.game.get_side() == "red" else self.var_black.get()

    def _legal_strings(self) -> set[str]:
        return {f"{a}{b}-{c}{d}" for (a, b, c, d) in self.game.legal_moves_iccs()}

    def _apply_move(self, mv: str) -> None:
        x1, y1, x2, y2 = parse_move_squares(mv)
        self.last_move = (x1, y1, x2, y2)
        self.game.make_move(mv)
        self.sel_from = None
        self._refresh_board()
        self._check_terminal()
        self.master.after(80, self._maybe_schedule_ai)

    def _check_terminal(self) -> None:
        legals = self.game.legal_moves_iccs()
        if legals:
            return
        from my_elephant.chess.board_utils import chess_board_from_base

        cb = chess_board_from_base(self.game.bb)
        if cb.is_checkmate():
            loser = "红方" if self.game.get_side() == "red" else "黑方"
            messagebox.showinfo("终局", f"{loser} 被将死，对局结束。")
        else:
            messagebox.showinfo("终局", "无子可动，和棋。")

    def _on_click(self, ev: tk.Event) -> None:
        if self._ai_busy:
            return
        if self._strategy_for_current() != STRATEGY_HUMAN:
            return
        pos = self._pixel_to_iccs(ev.x, ev.y)
        if pos is None:
            return
        ix, iy = pos
        arr = self._raw_board()
        ch = arr[iy, ix]
        turn = self.game.get_side()

        if self.sel_from is None:
            ps = _piece_side(str(ch) if ch else None)
            if ps == turn:
                self.sel_from = (ix, iy)
                self._refresh_board()
            return

        fx, fy = self.sel_from
        mv = f"{fx}{fy}-{ix}{iy}"
        if mv not in self._legal_strings():
            if _piece_side(str(ch) if ch else None) == turn:
                self.sel_from = (ix, iy)
            else:
                self.sel_from = None
            self._refresh_board()
            return

        self._apply_move(mv)

    def _maybe_schedule_ai(self) -> None:
        if self._ai_busy:
            return
        s = self._strategy_for_current()
        if s == STRATEGY_HUMAN:
            return
        legals = self.game.legal_moves_iccs()
        if not legals:
            return
        self._ai_busy = True
        self.status.config(text="AI 思考中…")

        def worker() -> None:
            try:
                g = copy_gameplay(self.game)
                if s == STRATEGY_NEURAL:
                    mv = _neural_pick_move(g, self.model, self.device, self.flist)
                else:

                    def ev(gp: GamePlay):
                        return eval_policy_value_at_root(gp, self.model, self.device, self.flist)

                    mv = mcts_search(g, ev, n_simulations=self.mcts_sims, c_puct=self.c_puct)
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda err=err: self._ai_done_error(err))
                return
            self.master.after(0, lambda m=mv: self._ai_done_move(m))

        threading.Thread(target=worker, daemon=True).start()

    def _ai_done_error(self, msg: str) -> None:
        self._ai_busy = False
        self.status.config(text="")
        messagebox.showerror("AI 错误", msg)
        self._refresh_board()

    def _ai_done_move(self, mv: str) -> None:
        self._ai_busy = False
        if mv not in self._legal_strings():
            self._refresh_board()
            messagebox.showerror("AI", f"非法着法: {mv}")
            return
        self._apply_move(mv)

    def _new_game(self) -> None:
        if self._ai_busy:
            return
        self.game = GamePlay()
        self.sel_from = None
        self.last_move = None
        self._refresh_board()
        self._maybe_schedule_ai()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tkinter 图形对弈（策略 / MCTS+双头）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0, help="-1 为 CPU")
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--mcts-sims", type=int, default=320, help="MCTS 模拟次数")
    p.add_argument("--c-puct", type=float, default=1.5, help="PUCT 探索系数")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    ckpt = torch_load_checkpoint(args.checkpoint, device)
    sd = ckpt["model"]
    num_res = int(ckpt.get("num_res_layers", 4))
    in_ch = int(
        ckpt.get("in_channels", ckpt.get("select_in_channels", args.in_channels or POLICY_SELECT_IN_CHANNELS))
    )
    filters = int(ckpt.get("filters", _infer_filters_from_state(sd)))
    model = SuccessorPolicy(num_res_layers=num_res, in_channels=in_ch, filters=filters).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    flist: dict[str, list[str]] = {
        "red": list(FEATURE_LIST["red"]),
        "black": list(FEATURE_LIST["black"]),
    }

    root = tk.Tk()
    app = XiangqiTkApp(root, model, device, flist, mcts_sims=args.mcts_sims, c_puct=args.c_puct)
    root.after(200, app._maybe_schedule_ai)
    root.mainloop()


if __name__ == "__main__":
    main()
