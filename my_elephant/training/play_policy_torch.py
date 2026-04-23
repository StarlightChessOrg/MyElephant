"""
Tkinter 图形对弈：圆形棋子、鼠标走子、对手上一手箭头提示；
红/黑可分别选「人类」「纯网络」「MCTS+策略价值网络」。
纯网络默认在根节点做 1 层 prior+价值搜索（见 ``infer_1ply_value_prior_move``），可用 ``--neural-mode greedy`` 恢复旧贪心。
"""
from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import torch

from my_elephant.chess import GamePlay
from my_elephant.chess.features import parse_move_squares

from my_elephant.training.mcts_engine import MCTSSearchStats, copy_gameplay, mcts_search
from my_elephant.training.play_model_loader import load_successor_policy_for_play
from my_elephant.training.policy_eval_http import (
    PolicyHTTPEvalClient,
    make_http_evaluator,
    spawn_mcts_http_eval_cluster,
    terminate_http_eval_cluster,
)
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    eval_policy_value_at_root,
    infer_1ply_value_prior_move,
    infer_greedy_move_string,
)

STRATEGY_HUMAN = "人类"
STRATEGY_NEURAL = "纯网络"
STRATEGY_MCTS = "MCTS+策略价值网络"
STRATEGIES = (STRATEGY_HUMAN, STRATEGY_NEURAL, STRATEGY_MCTS)
# ttk.Combobox 的 width 为字符宽度；略大于最长项以免「MCTS+策略价值网络」被裁切。
_STRATEGY_COMBO_WIDTH = max(22, max(len(s) for s in STRATEGIES) + 6)
# 状态栏 wraplength（像素）与固定字符宽度：避免「AI 思考中」短文案时右栏变窄、MCTS 结束后变宽。
_STATUS_WRAPLENGTH_PX = 280
# ttk.Label 的 width 为「平均字符」数，略大于 wrap 所需以免高 DPI 下仍抖动。
_STATUS_LABEL_WIDTH_CHARS = 30

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


def _piece_side(ch: str | None) -> str | None:
    if not ch:
        return None
    return "red" if ch.isupper() else "black"


def _neural_pick_move(gp: GamePlay, model: SuccessorPolicy, device: torch.device, flist: dict) -> str:
    mv = infer_greedy_move_string(gp, model, device, flist)
    legals = {f"{a}{b}-{c}{d}" for (a, b, c, d) in gp.legal_moves_iccs()}
    if mv not in legals:
        raise RuntimeError(f"网络贪心着法不在合法集中: {mv!r}")
    return mv


class XiangqiTkApp:
    def __init__(
        self,
        master: tk.Tk,
        model: SuccessorPolicy,
        device: torch.device,
        flist: dict[str, list[str]],
        mcts_sims: int,
        c_puct: float,
        mcts_max_seconds: float | None = None,
        *,
        mcts_workers: int | None = None,
        mcts_virtual_loss: float = 3.0,
        mcts_http_client: PolicyHTTPEvalClient | None = None,
        mcts_http_procs: list[subprocess.Popen] | None = None,
        neural_mode: str = "1ply",
        neural_prior_weight: float = 1.0,
        neural_value_weight: float = 1.0,
    ) -> None:
        self.master = master
        self.model = model
        self.device = device
        self.flist = flist
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.mcts_max_seconds = mcts_max_seconds
        self.mcts_workers = mcts_workers
        self.mcts_virtual_loss = mcts_virtual_loss
        self._mcts_http_client = mcts_http_client
        self._mcts_http_procs = list(mcts_http_procs) if mcts_http_procs else []
        self.neural_mode = neural_mode
        self.neural_prior_weight = neural_prior_weight
        self.neural_value_weight = neural_value_weight
        self.game = GamePlay()
        self.sel_from: tuple[int, int] | None = None
        self.last_move: tuple[int, int, int, int] | None = None
        self._ai_busy = False
        self._last_mcts_info: str | None = None

        nw_cap = mcts_workers if mcts_workers is not None else (os.cpu_count() or 1)
        nw_cap = max(1, min(64, nw_cap))
        self._mcts_executor: ThreadPoolExecutor | None = None
        if nw_cap > 1:
            self._mcts_executor = ThreadPoolExecutor(max_workers=nw_cap, thread_name_prefix="mcts")

        master.title("MyElephant 象棋对弈")
        self.CELL = 52
        self.OFF = 36
        # 交点坐标系：棋子落在 (OFF+ix*CELL, OFF+iy*CELL)，最外交点为 (8,9)，圆半径需留边。
        _r = self.CELL // 2 - 4
        _pad = 6
        self.CW = int(self.OFF * 2 + 8 * self.CELL + _r + _pad)
        self.CH = int(self.OFF * 2 + 9 * self.CELL + _r + _pad)

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
        cb_r = ttk.Combobox(
            right, textvariable=self.var_red, values=STRATEGIES, state="readonly", width=_STRATEGY_COMBO_WIDTH
        )
        cb_r.pack(anchor=tk.W, pady=(0, 8))
        cb_r.bind("<<ComboboxSelected>>", lambda _e: master.after(50, self._maybe_schedule_ai))

        ttk.Label(right, text="黑方策略").pack(anchor=tk.W)
        self.var_black = tk.StringVar(value=STRATEGY_NEURAL)
        cb_b = ttk.Combobox(
            right, textvariable=self.var_black, values=STRATEGIES, state="readonly", width=_STRATEGY_COMBO_WIDTH
        )
        cb_b.pack(anchor=tk.W, pady=(0, 8))
        cb_b.bind("<<ComboboxSelected>>", lambda _e: master.after(50, self._maybe_schedule_ai))

        ttk.Button(right, text="新局", command=self._new_game).pack(anchor=tk.W, pady=4)
        self.status = ttk.Label(
            right,
            text="",
            wraplength=_STATUS_WRAPLENGTH_PX,
            width=_STATUS_LABEL_WIDTH_CHARS,
            anchor=tk.W,
        )
        self.status.pack(anchor=tk.W, pady=8)

        self._draw_static_grid()
        self._refresh_board()

        master.protocol("WM_DELETE_WINDOW", self._on_close_window)

    def _shutdown_mcts_http_workers(self) -> None:
        """终止多进程 HTTP 评估子进程（可重复调用）。"""
        procs, self._mcts_http_procs = self._mcts_http_procs, []
        if procs:
            terminate_http_eval_cluster(procs)

    def _shutdown_mcts_executor(self) -> None:
        """关闭 MCTS 常驻线程池（窗口关闭或进程退出时调用；可重复调用）。"""
        ex, self._mcts_executor = self._mcts_executor, None
        if ex is None:
            return
        try:
            ex.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def _shutdown_all_mcts_resources(self) -> None:
        """先停 MCTS 线程池（等飞行中的 HTTP 评估结束），再关 HTTP 子进程。"""
        self._shutdown_mcts_executor()
        self._shutdown_mcts_http_workers()

    def _on_close_window(self) -> None:
        self._shutdown_all_mcts_resources()
        self.master.destroy()

    def _iccs_center(self, ix: int, iy: int) -> tuple[float, float]:
        """ICCS 交叉点 (ix,iy) 的像素坐标（棋子中心落在纵线与横线交点上，非方格中心）。"""
        cx = float(self.OFF + ix * self.CELL)
        cy = float(self.OFF + iy * self.CELL)
        return cx, cy

    def _draw_traverse_mark(self, cx: float, cy: float) -> None:
        """炮位、卒林等传统「花心」：交叉短线。"""
        h = max(6, self.CELL // 7)
        c = self.canvas
        w, t = "#4a3225", "grid"
        c.create_line(cx - h, cy, cx + h, cy, fill=w, width=1, tags=t)
        c.create_line(cx, cy - h, cx, cy + h, fill=w, width=1, tags=t)

    def _draw_static_grid(self) -> None:
        """传统棋盘：外框、10 横 9 纵（河界断竖线）、九宫斜线、炮位/卒林花心、楚河汉界。"""
        c = self.canvas
        o = self.OFF
        u = self.CELL
        x0, x1 = o, o + 8 * u
        y0, y1 = o, o + 9 * u
        col = "#4a3225"
        t = "grid"
        # 河界：第 5、6 条横线之间无纵线（iy=4 与 iy=5 之间）
        y_river_top = o + 4 * u
        y_river_bot = o + 5 * u

        c.create_rectangle(x0, y0, x1, y1, outline=col, width=2, tags=t)
        for i in range(10):
            y = o + i * u
            c.create_line(x0, y, x1, y, fill=col, width=1, tags=t)
        for i in range(9):
            x = o + i * u
            if i in (0, 8):
                # 边线纵贯河界（与传统木盘一致）
                c.create_line(x, y0, x, y1, fill=col, width=1, tags=t)
            else:
                c.create_line(x, y0, x, y_river_top, fill=col, width=1, tags=t)
                c.create_line(x, y_river_bot, x, y1, fill=col, width=1, tags=t)

        def line_ij(ix_a: int, iy_a: int, ix_b: int, iy_b: int) -> None:
            ax, ay = self._iccs_center(ix_a, iy_a)
            bx, by = self._iccs_center(ix_b, iy_b)
            c.create_line(ax, ay, bx, by, fill=col, width=1, tags=t)

        # 黑方九宫（顶中 iy 0–2）
        line_ij(3, 0, 5, 2)
        line_ij(5, 0, 3, 2)
        # 红方九宫（底中 iy 7–9）
        line_ij(3, 7, 5, 9)
        line_ij(5, 7, 3, 9)

        for ix, iy in ((1, 2), (7, 2), (1, 7), (7, 7)):
            self._draw_traverse_mark(*self._iccs_center(ix, iy))
        for ix in (0, 2, 4, 6, 8):
            self._draw_traverse_mark(*self._iccs_center(ix, 3))
        for ix in (0, 2, 4, 6, 8):
            self._draw_traverse_mark(*self._iccs_center(ix, 6))

        ry = o + 4.5 * u
        font = ("Microsoft YaHei", 13, "bold")
        c.create_text(o + 2 * u, ry, text="楚 河", fill="#5d4037", font=font, tags=t)
        c.create_text(o + 6 * u, ry, text="漢 界", fill="#5d4037", font=font, tags=t)

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
            d = max(self.CELL // 3, r + 2)
            self.canvas.create_rectangle(
                cx - d, cy - d, cx + d, cy + d, outline="#ffeb3b", width=3, tags="selhi"
            )

        side = "红方" if self.game.get_side() == "red" else "黑方"
        head = f"轮到 {side} 走棋"
        if self._last_mcts_info:
            self.status.config(text=f"{head}\n{self._last_mcts_info}")
        else:
            self.status.config(text=head)

    def _pixel_to_iccs(self, px: float, py: float) -> tuple[int, int] | None:
        """点击映射到最近的交叉点（与棋子落点一致）。"""
        fx = (px - self.OFF) / self.CELL
        fy = (py - self.OFF) / self.CELL
        ix = int(round(fx))
        iy = int(round(fy))
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
            mcts_st: MCTSSearchStats | None = None
            try:
                g = copy_gameplay(self.game)
                if s == STRATEGY_NEURAL:
                    if self.neural_mode == "greedy":
                        mv = _neural_pick_move(g, self.model, self.device, self.flist)
                    else:
                        mv = infer_1ply_value_prior_move(
                            g,
                            self.model,
                            self.device,
                            self.flist,
                            prior_weight=self.neural_prior_weight,
                            value_weight=self.neural_value_weight,
                        )
                else:
                    if self._mcts_http_client is not None:
                        ev = make_http_evaluator(self._mcts_http_client)
                    else:

                        def ev(gp: GamePlay):
                            return eval_policy_value_at_root(gp, self.model, self.device, self.flist)

                    mv, mcts_st = mcts_search(
                        g,
                        ev,
                        n_simulations=self.mcts_sims,
                        c_puct=self.c_puct,
                        max_seconds=self.mcts_max_seconds,
                        virtual_loss=self.mcts_virtual_loss,
                        n_workers=self.mcts_workers,
                        thread_pool=self._mcts_executor,
                    )
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda err=err: self._ai_done_error(err))
                return
            if mcts_st is not None:
                self.master.after(0, lambda m=mv, st=mcts_st: self._ai_done_move(m, st))
            else:
                self.master.after(0, lambda m=mv: self._ai_done_move(m))

        threading.Thread(target=worker, daemon=True).start()

    def _ai_done_error(self, msg: str) -> None:
        self._ai_busy = False
        self.status.config(text="")
        messagebox.showerror("AI 错误", msg)
        self._refresh_board()

    def _ai_done_move(self, mv: str, mcts_st: MCTSSearchStats | None = None) -> None:
        self._ai_busy = False
        if mcts_st is not None:
            tlim = (
                f"{mcts_st.requested_max_seconds:.2f}s"
                if mcts_st.requested_max_seconds is not None
                else "—"
            )
            self._last_mcts_info = (
                f"MCTS 玩法{mcts_st.n_playouts}/{mcts_st.requested_simulations} "
                f"墙钟{mcts_st.elapsed_seconds:.3f}s 时限{tlim}\n"
                f"网络展开{mcts_st.n_expansions} 根访问{mcts_st.root_total_visits} "
                f"停止={mcts_st.stopped_by} "
                f"并行{mcts_st.parallel_workers}线程 VL={mcts_st.virtual_loss:g}"
            )
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
        self._last_mcts_info = None
        self._refresh_board()
        self._maybe_schedule_ai()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tkinter 图形对弈（策略 / MCTS+策略价值）")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="已忽略：本对弈程序固定使用 CPU 加载权重与推理（小模型避免 GPU 往返开销）",
    )
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument(
        "--mcts-sims",
        type=int,
        default=192,
        help="MCTS 模拟次数上限（默认较保守以缩短墙钟；要强棋可加大或配合 --mcts-max-seconds）",
    )
    p.add_argument(
        "--mcts-max-seconds",
        type=float,
        default=5.0,
        help="MCTS 墙钟时间上限（秒），默认 5；与 --mcts-sims 先到先停；<=0 表示不限时",
    )
    p.add_argument(
        "--mcts-workers",
        type=int,
        default=None,
        help="MCTS 并行模拟线程数，默认等于 CPU 逻辑核心数；1 为单线程（无并行）",
    )
    p.add_argument(
        "--mcts-virtual-loss",
        type=float,
        default=3.0,
        help="多线程 MCTS 虚拟损失系数（仅 workers>1 时生效）",
    )
    p.add_argument(
        "--mcts-http-workers",
        type=int,
        default=0,
        help=">0 时启动等量本机 HTTP 子进程做策略评估（127.0.0.1），父进程 MCTS 经 HTTP 分发以利用多核 CPU",
    )
    p.add_argument(
        "--mcts-http-base-port",
        type=int,
        default=17890,
        help="HTTP 评估子进程监听端口起始（每进程 +1）",
    )
    p.add_argument("--c-puct", type=float, default=1.5, help="PUCT 探索系数")
    p.add_argument(
        "--neural-mode",
        type=str,
        default="1ply",
        choices=("1ply", "greedy"),
        help="纯网络：1ply=根枚举 + prior 与后继价值头单层打分；greedy=原两阶段 argmax",
    )
    p.add_argument(
        "--neural-prior-weight",
        type=float,
        default=1.0,
        help="纯网络 1ply：log prior 项系数",
    )
    p.add_argument(
        "--neural-value-weight",
        type=float,
        default=1.0,
        help="纯网络 1ply：-V(后继) 项系数",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    mcts_max_seconds = None if args.mcts_max_seconds is not None and args.mcts_max_seconds <= 0 else float(args.mcts_max_seconds)
    _ = args.gpu  # 保留 CLI 兼容，对弈固定 CPU
    device = torch.device("cpu")

    mcts_workers_eff = args.mcts_workers
    if int(args.mcts_http_workers) > 0:
        if mcts_workers_eff is None:
            mcts_workers_eff = min(64, max(int(args.mcts_http_workers), os.cpu_count() or 1))
        elif mcts_workers_eff <= 1:
            mcts_workers_eff = max(2, min(int(args.mcts_http_workers), os.cpu_count() or 2))

    mcts_http_procs: list[subprocess.Popen] = []
    mcts_http_client: PolicyHTTPEvalClient | None = None
    if int(args.mcts_http_workers) > 0:
        mcts_http_procs, http_urls = spawn_mcts_http_eval_cluster(
            args.checkpoint,
            int(args.mcts_http_workers),
            int(args.mcts_http_base_port),
            in_channels=args.in_channels,
        )
        mcts_http_client = PolicyHTTPEvalClient(http_urls)

    model, flist = load_successor_policy_for_play(
        args.checkpoint, device, in_channels=args.in_channels
    )

    root = tk.Tk()
    app = XiangqiTkApp(
        root,
        model,
        device,
        flist,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
        mcts_max_seconds=mcts_max_seconds,
        mcts_workers=mcts_workers_eff,
        mcts_virtual_loss=args.mcts_virtual_loss,
        mcts_http_client=mcts_http_client,
        mcts_http_procs=mcts_http_procs,
        neural_mode=args.neural_mode,
        neural_prior_weight=args.neural_prior_weight,
        neural_value_weight=args.neural_value_weight,
    )
    atexit.register(app._shutdown_all_mcts_resources)
    root.after(200, app._maybe_schedule_ai)
    root.mainloop()


if __name__ == "__main__":
    main()
