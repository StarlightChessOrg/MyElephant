"""
网页版图形对弈：浏览器访问棋盘、点击走子；红/黑可选「人类」「纯网络」「MCTS+策略价值网络」。
纯网络默认在根节点做 1 层 prior+价值搜索（见 ``infer_1ply_value_prior_move``），可用 ``--neural-mode greedy`` 恢复旧贪心。

无显示器服务器上运行示例::

    python -m my_elephant.training.play_policy_torch --checkpoint /path/to.ckpt --host 0.0.0.0 --port 8765

本机浏览器打开 ``http://<服务器IP>:8765`` 即可对弈。依赖 ``flask``（见 ``pyproject.toml`` 的 ``play`` 可选依赖）。
"""
from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from my_elephant.chess import GamePlay
from my_elephant.chess.features import parse_move_squares

from my_elephant.training.mcts_engine import (
    MCTSSearchStats,
    copy_gameplay,
    descend_mcts_subtree,
    mcts_search,
)
from my_elephant.training.play_model_loader import load_successor_policy_for_play
from my_elephant.training.policy_eval_http import (
    PolicyHTTPEvalClient,
    make_http_evaluator,
    spawn_mcts_http_eval_cluster,
    terminate_http_eval_cluster,
)
from my_elephant.training.policy_torch import (
    QueuedBatchedRootEvaluator,
    SuccessorPolicy,
    eval_policy_value_at_root,
    infer_1ply_value_prior_move,
    infer_greedy_move_string,
)

STRATEGY_HUMAN = "人类"
STRATEGY_NEURAL = "纯网络"
STRATEGY_MCTS = "MCTS+策略价值网络"
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


def _select_play_device(gpu: int) -> torch.device:
    """对弈推理：``gpu>=0`` 且可用 CUDA 时用 ``cuda:gpu``，否则 CPU（``gpu<0`` 为强制 CPU）。"""
    if gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("[play] 未检测到 CUDA，推理使用 CPU", flush=True)
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


class XiangqiPlaySession:
    """对弈状态与 AI/MCTS 逻辑（无 UI），供 Flask 路由在锁内调用。"""

    def __init__(
        self,
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
        self._lock = threading.Lock()
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
        self._root_eval_batcher: QueuedBatchedRootEvaluator | None = None
        if mcts_http_client is None and device.type == "cuda":
            self._root_eval_batcher = QueuedBatchedRootEvaluator(model, device, flist)
        self.game = GamePlay()
        self.sel_from: tuple[int, int] | None = None
        self.last_move: tuple[int, int, int, int] | None = None
        self._ai_busy = False
        self._last_mcts_info: str | None = None
        self._mcts_prev_tree_root: object | None = None
        self._mcts_moves_since_search: list[str] = []
        self.strategy_red = STRATEGY_HUMAN
        self.strategy_black = STRATEGY_NEURAL
        self._toasts: list[dict[str, str]] = []
        self._ai_errors: list[str] = []

        nw_cap = mcts_workers if mcts_workers is not None else (os.cpu_count() or 1)
        nw_cap = max(1, min(64, nw_cap))
        self._mcts_executor: ThreadPoolExecutor | None = None
        if nw_cap > 1:
            self._mcts_executor = ThreadPoolExecutor(max_workers=nw_cap, thread_name_prefix="mcts")

    def _invalidate_mcts_subtree_cache(self) -> None:
        self._mcts_prev_tree_root = None
        self._mcts_moves_since_search.clear()

    def _shutdown_mcts_http_workers(self) -> None:
        procs, self._mcts_http_procs = self._mcts_http_procs, []
        if procs:
            terminate_http_eval_cluster(procs)

    def _shutdown_mcts_executor(self) -> None:
        ex, self._mcts_executor = self._mcts_executor, None
        if ex is None:
            return
        try:
            ex.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    def _shutdown_root_eval_batcher(self) -> None:
        b, self._root_eval_batcher = self._root_eval_batcher, None
        if b is not None:
            b.close()

    def shutdown_all_mcts_resources(self) -> None:
        self._shutdown_mcts_executor()
        self._shutdown_root_eval_batcher()
        self._shutdown_mcts_http_workers()

    def _raw_board(self) -> np.ndarray:
        return np.asarray(self.game.bb._board[::-1])

    def _status_head(self) -> str:
        if self._ai_busy:
            return "AI 思考中…"
        side = "红方" if self.game.get_side() == "red" else "黑方"
        return f"轮到 {side} 走棋"

    def _strategy_for_current(self) -> str:
        return self.strategy_red if self.game.get_side() == "red" else self.strategy_black

    def _legal_strings(self) -> set[str]:
        return {f"{a}{b}-{c}{d}" for (a, b, c, d) in self.game.legal_moves_iccs()}

    def _check_terminal_unlocked(self) -> None:
        legals = self.game.legal_moves_iccs()
        if legals:
            return
        from my_elephant.chess.board_utils import chess_board_from_base

        cb = chess_board_from_base(self.game.bb)
        if cb.is_checkmate():
            loser = "红方" if self.game.get_side() == "red" else "黑方"
            self._toasts.append({"kind": "info", "title": "终局", "body": f"{loser} 被将死，对局结束。"})
        else:
            self._toasts.append({"kind": "info", "title": "终局", "body": "无子可动，和棋。"})

    def _apply_move_unlocked(self, mv: str) -> None:
        if self._mcts_prev_tree_root is not None:
            self._mcts_moves_since_search.append(mv)
        x1, y1, x2, y2 = parse_move_squares(mv)
        self.last_move = (x1, y1, x2, y2)
        self.game.make_move(mv)
        self.sel_from = None
        self._check_terminal_unlocked()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict:
        arr = self._raw_board()
        rows: list[list[dict[str, str | None]]] = []
        for iy in range(10):
            row: list[dict[str, str | None]] = []
            for ix in range(9):
                ch = arr[iy, ix]
                if not ch:
                    row.append({"ch": None, "side": None, "label": ""})
                else:
                    s = str(ch)
                    row.append({"ch": s, "side": _piece_side(s), "label": _PIECE_CHAR.get(s, "?")})
            rows.append(row)
        head = self._status_head()
        status_text = head if not self._last_mcts_info else f"{head}\n{self._last_mcts_info}"
        return {
            "board": rows,
            "side_to_move": self.game.get_side(),
            "sel_from": list(self.sel_from) if self.sel_from else None,
            "last_move": list(self.last_move) if self.last_move else None,
            "strategy_red": self.strategy_red,
            "strategy_black": self.strategy_black,
            "strategies": list(STRATEGIES),
            "status_text": status_text,
            "ai_busy": self._ai_busy,
            "current_strategy": self._strategy_for_current(),
        }

    def pop_client_messages(self) -> dict:
        """取走待弹窗的终局提示与 AI 错误（各请求消费一次）。"""
        with self._lock:
            toasts = self._toasts
            errs = self._ai_errors
            self._toasts = []
            self._ai_errors = []
            return {"toasts": toasts, "ai_errors": errs}

    def set_strategies(self, red: str, black: str) -> dict | None:
        if red not in STRATEGIES or black not in STRATEGIES:
            return {"error": "非法策略"}
        with self._lock:
            if self._ai_busy:
                return {"error": "AI 思考中"}
            self.strategy_red = red
            self.strategy_black = black
        self.maybe_schedule_ai()
        return None

    def new_game(self) -> dict | None:
        with self._lock:
            if self._ai_busy:
                return {"error": "AI 思考中"}
            self.game = GamePlay()
            self.sel_from = None
            self.last_move = None
            self._last_mcts_info = None
            self._invalidate_mcts_subtree_cache()
        self.maybe_schedule_ai()
        return None

    def click_cell(self, ix: int, iy: int) -> dict | None:
        with self._lock:
            if self._ai_busy:
                return {"error": "AI 思考中"}
            if self._strategy_for_current() != STRATEGY_HUMAN:
                return {"error": "当前非人类行棋"}
            if not (0 <= ix <= 8 and 0 <= iy <= 9):
                return {"error": "坐标越界"}
            arr = self._raw_board()
            ch = arr[iy, ix]
            turn = self.game.get_side()

            if self.sel_from is None:
                ps = _piece_side(str(ch) if ch else None)
                if ps == turn:
                    self.sel_from = (ix, iy)
                return None

            fx, fy = self.sel_from
            mv = f"{fx}{fy}-{ix}{iy}"
            if mv not in self._legal_strings():
                if _piece_side(str(ch) if ch else None) == turn:
                    self.sel_from = (ix, iy)
                else:
                    self.sel_from = None
                return None

            self._apply_move_unlocked(mv)

        self.maybe_schedule_ai()
        return None

    def maybe_schedule_ai(self) -> None:
        with self._lock:
            if self._ai_busy:
                return
            s = self._strategy_for_current()
            if s == STRATEGY_HUMAN:
                return
            legals = self.game.legal_moves_iccs()
            if not legals:
                return
            self._ai_busy = True

            if s == STRATEGY_MCTS:
                moves_chain = list(self._mcts_moves_since_search)
                self._mcts_moves_since_search.clear()
                reuse_sub = descend_mcts_subtree(self._mcts_prev_tree_root, moves_chain, self.game)
            else:
                reuse_sub = None

            game_snap = copy_gameplay(self.game)
            model = self.model
            device = self.device
            flist = self.flist
            neural_mode = self.neural_mode
            neural_pw = self.neural_prior_weight
            neural_vw = self.neural_value_weight
            mcts_sims = self.mcts_sims
            c_puct = self.c_puct
            mcts_max_seconds = self.mcts_max_seconds
            mcts_vl = self.mcts_virtual_loss
            mcts_workers = self.mcts_workers
            ex = self._mcts_executor
            http_client = self._mcts_http_client
            root_batcher = self._root_eval_batcher

        def worker() -> None:
            mcts_st: MCTSSearchStats | None = None
            root_out: object | None = None
            try:
                g = copy_gameplay(game_snap)
                if s == STRATEGY_NEURAL:
                    if neural_mode == "greedy":
                        mv = _neural_pick_move(g, model, device, flist)
                    else:
                        mv = infer_1ply_value_prior_move(
                            g,
                            model,
                            device,
                            flist,
                            prior_weight=neural_pw,
                            value_weight=neural_vw,
                        )
                else:
                    if http_client is not None:
                        ev = make_http_evaluator(http_client)
                    elif root_batcher is not None:
                        ev = root_batcher.eval_sync
                    else:
                        ev = lambda gp: eval_policy_value_at_root(gp, model, device, flist)

                    mv, mcts_st, root_out = mcts_search(
                        g,
                        ev,
                        n_simulations=mcts_sims,
                        c_puct=c_puct,
                        max_seconds=mcts_max_seconds,
                        virtual_loss=mcts_vl,
                        n_workers=mcts_workers,
                        thread_pool=ex,
                        reuse_subtree=reuse_sub,
                    )
            except Exception as e:
                err = str(e)
                with self._lock:
                    self._ai_busy = False
                    self._invalidate_mcts_subtree_cache()
                    self._ai_errors.append(err)
                return

            with self._lock:
                self._ai_busy = False
                if mcts_st is None:
                    self._invalidate_mcts_subtree_cache()
                else:
                    tlim = (
                        f"{mcts_st.requested_max_seconds:.2f}s"
                        if mcts_st.requested_max_seconds is not None
                        else "—"
                    )
                    self._last_mcts_info = (
                        f"MCTS 玩法{mcts_st.n_playouts}/{mcts_st.requested_simulations} "
                        f"墙钟{mcts_st.elapsed_seconds:.3f}s 时限{tlim}\n"
                        f"网络展开{mcts_st.n_expansions} 根访问{mcts_st.root_total_visits}\n"
                        f"停止={mcts_st.stopped_by}\n"
                        f"并行{mcts_st.parallel_workers}线程 VL={mcts_st.virtual_loss:g}"
                    )
                legals_now = self._legal_strings()
                if mv not in legals_now:
                    self._invalidate_mcts_subtree_cache()
                    self._ai_errors.append(f"非法着法: {mv}")
                    return
                if mcts_st is not None and root_out is not None:
                    self._mcts_prev_tree_root = root_out
                self._apply_move_unlocked(mv)

            self.maybe_schedule_ai()

        threading.Thread(target=worker, daemon=True).start()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="网页版图形对弈（策略 / MCTS+策略价值），便于无头服务器部署")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="推理设备：>=0 且存在 CUDA 时使用 ``cuda:N``；无 CUDA 时自动回退 CPU；``-1`` 强制 CPU",
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
    p.add_argument("--host", type=str, default="127.0.0.1", help="监听地址；服务器对外访问请用 0.0.0.0")
    p.add_argument("--port", type=int, default=8765, help="监听端口")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Flask debug（仅本机调试；生产勿开）",
    )
    return p.parse_args()


def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MyElephant 象棋对弈</title>
  <style>
    :root {
      --bg: #2c2416;
      --panel: #3e3426;
      --line: #4a3225;
      --cell: #e8d4b8;
      --red: #c62828;
      --black: #1565c0;
      --text: #efe7dc;
      --accent: #ff9800;
      --sel: #ffeb3b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
      display: flex; flex-wrap: wrap; gap: 16px; padding: 16px; justify-content: center;
    }
    .board-wrap {
      background: var(--cell); border-radius: 8px; padding: 12px;
      box-shadow: 0 8px 24px rgba(0,0,0,.45);
    }
    .board {
      display: grid; grid-template-columns: repeat(9, 44px); grid-template-rows: repeat(10, 44px);
      position: relative; width: calc(9 * 44px); height: calc(10 * 44px);
      background: linear-gradient(to bottom, #f0e6d4 0%, #e8d4b8 100%);
    }
    .cell {
      border: 1px solid rgba(74,50,37,.25); cursor: pointer; position: relative;
      display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: bold;
      user-select: none;
    }
    .cell:hover { background: rgba(255,255,255,.12); }
    .piece-red { color: #fff; background: var(--red); border-radius: 50%; width: 38px; height: 38px;
      display: flex; align-items: center; justify-content: center; border: 2px solid #3e2723; }
    .piece-black { color: #fff; background: var(--black); border-radius: 50%; width: 38px; height: 38px;
      display: flex; align-items: center; justify-content: center; border: 2px solid #3e2723; }
    .sel { outline: 3px solid var(--sel); outline-offset: -2px; border-radius: 4px; }
    .last-from, .last-to { box-shadow: inset 0 0 0 2px var(--accent); }
    .sidepanel {
      min-width: 280px; max-width: 380px; background: var(--panel); padding: 16px; border-radius: 8px;
    }
    label { display: block; margin-top: 8px; font-size: 14px; opacity: .9; }
    select { width: 100%; padding: 8px; margin-top: 4px; border-radius: 6px; border: 1px solid #5d4e3a;
      background: #2a2218; color: var(--text); font-size: 14px; }
    button {
      margin-top: 12px; padding: 10px 16px; border: none; border-radius: 6px;
      background: #6d4c41; color: #fff; font-size: 15px; cursor: pointer; width: 100%;
    }
    button:hover { background: #8d6e63; }
    #status {
      margin-top: 14px; white-space: pre-wrap; font-size: 13px; line-height: 1.45;
      padding: 10px; background: rgba(0,0,0,.2); border-radius: 6px; min-height: 4em;
    }
    h1 { font-size: 1.1rem; margin: 0 0 12px; font-weight: 600; }
  </style>
</head>
<body>
  <div class="board-wrap">
    <div class="board" id="board"></div>
  </div>
  <div class="sidepanel">
    <h1>MyElephant 象棋对弈</h1>
    <label>红方策略</label>
    <select id="sel-red"></select>
    <label>黑方策略</label>
    <select id="sel-black"></select>
    <button type="button" id="btn-new">新局</button>
    <div id="status"></div>
  </div>
<script>
(function () {
  const boardEl = document.getElementById("board");
  const statusEl = document.getElementById("status");
  const selRed = document.getElementById("sel-red");
  const selBlack = document.getElementById("sel-black");
  const btnNew = document.getElementById("btn-new");

  function showAlert(title, body) {
    alert(title + "\\n\\n" + body);
  }

  function renderBoard(snap) {
    boardEl.innerHTML = "";
    const strategies = snap.strategies || [];
    if (selRed.options.length === 0) {
      strategies.forEach(function (t) {
        const o = document.createElement("option"); o.value = t; o.textContent = t; selRed.appendChild(o);
      });
      strategies.forEach(function (t) {
        const o = document.createElement("option"); o.value = t; o.textContent = t; selBlack.appendChild(o);
      });
    }
    selRed.value = snap.strategy_red;
    selBlack.value = snap.strategy_black;
    statusEl.textContent = snap.status_text || "";

    const lm = snap.last_move;
    const sf = snap.sel_from;
    const busy = !!snap.ai_busy;
    for (let iy = 0; iy < 10; iy++) {
      for (let ix = 0; ix < 9; ix++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.ix = ix; cell.dataset.iy = iy;
        if (lm && ix === lm[0] && iy === lm[1]) cell.classList.add("last-from");
        if (lm && ix === lm[2] && iy === lm[3]) cell.classList.add("last-to");
        if (sf && ix === sf[0] && iy === sf[1]) cell.classList.add("sel");
        const sq = snap.board[iy][ix];
        if (sq.ch) {
          const span = document.createElement("span");
          span.textContent = sq.label || "?";
          span.className = sq.side === "red" ? "piece-red" : "piece-black";
          cell.appendChild(span);
        }
        cell.addEventListener("click", function () {
          if (busy) return;
          fetch("/api/click", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ix: ix, iy: iy })
          }).then(function (r) { return r.json(); }).then(function (j) {
            if (j.error) showAlert("走子", j.error);
          }).catch(function () {});
        });
        boardEl.appendChild(cell);
      }
    }
  }

  function schedulePoll() {
    fetch("/api/state")
      .then(function (r) { return r.json(); })
      .then(function (s) {
        renderBoard(s);
        const delay = s.ai_busy ? 400 : 900;
        return fetch("/api/messages").then(function (r) { return r.json(); }).then(function (msg) {
          (msg.toasts || []).forEach(function (t) {
            if (t.kind === "info") showAlert(t.title || "提示", t.body || "");
          });
          (msg.ai_errors || []).forEach(function (e) { showAlert("AI 错误", e); });
          return delay;
        });
      })
      .catch(function () { return 1200; })
      .then(function (delay) { setTimeout(schedulePoll, delay); });
  }

  selRed.addEventListener("change", function () {
    fetch("/api/strategies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ red: selRed.value, black: selBlack.value })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error) showAlert("策略", j.error);
    });
  });
  selBlack.addEventListener("change", function () {
    fetch("/api/strategies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ red: selRed.value, black: selBlack.value })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error) showAlert("策略", j.error);
    });
  });
  btnNew.addEventListener("click", function () {
    fetch("/api/new_game", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (j.error) showAlert("新局", j.error); });
  });

  schedulePoll();
})();
</script>
</body>
</html>"""


def main() -> None:
    try:
        from flask import Flask, jsonify, request
    except ImportError as e:
        raise SystemExit(
            "缺少依赖 flask。请安装：pip install 'flask>=2.3' 或 pip install -e '.[play]'"
        ) from e

    args = _parse_args()
    mcts_max_seconds = (
        None if args.mcts_max_seconds is not None and args.mcts_max_seconds <= 0 else float(args.mcts_max_seconds)
    )
    device = _select_play_device(int(args.gpu))
    print(f"[play] 推理设备: {device}", flush=True)

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
            gpu=int(args.gpu),
        )
        mcts_http_client = PolicyHTTPEvalClient(http_urls)

    model, flist = load_successor_policy_for_play(
        args.checkpoint, device, in_channels=args.in_channels
    )

    session = XiangqiPlaySession(
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
    atexit.register(session.shutdown_all_mcts_resources)

    app = Flask(__name__)

    @app.get("/")
    def index():
        from flask import Response

        return Response(_html_page(), mimetype="text/html; charset=utf-8")

    @app.get("/api/state")
    def api_state():
        return jsonify(session.snapshot())

    @app.get("/api/messages")
    def api_messages():
        return jsonify(session.pop_client_messages())

    @app.post("/api/click")
    def api_click():
        data = request.get_json(silent=True) or {}
        try:
            ix = int(data.get("ix", -1))
            iy = int(data.get("iy", -1))
        except (TypeError, ValueError):
            return jsonify({"error": "坐标无效"}), 400
        err = session.click_cell(ix, iy)
        return jsonify(err or {})

    @app.post("/api/strategies")
    def api_strategies():
        data = request.get_json(silent=True) or {}
        red = data.get("red")
        black = data.get("black")
        if not isinstance(red, str) or not isinstance(black, str):
            return jsonify({"error": "参数无效"}), 400
        err = session.set_strategies(red, black)
        return jsonify(err or {})

    @app.post("/api/new_game")
    def api_new_game():
        err = session.new_game()
        return jsonify(err or {})

    def _kick_ai():
        session.maybe_schedule_ai()

    threading.Timer(0.25, _kick_ai).start()

    print(
        f"[play] 网页对弈: http://{args.host}:{args.port}/  （远程访问请将 --host 设为 0.0.0.0）",
        flush=True,
    )
    app.run(host=args.host, port=int(args.port), debug=bool(args.debug), threaded=True)


if __name__ == "__main__":
    main()
