"""本地 HTTP 评估：父进程 MCTS 将局面 FEN POST 到子进程，子进程内跑 ``eval_policy_value_at_root``（优先 GPU，见 ``policy_eval_worker``）。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from my_elephant.chess.session import GamePlay
from my_elephant.training.policy_torch import SuccessorPolicy, eval_policy_value_at_root


def gameplay_to_fen(gp: GamePlay) -> str:
    return gp.bb.to_fen()


def gameplay_from_fen(fen: str) -> GamePlay:
    from cchess.board import BaseChessBoard
    from cchess.piece import ChessSide

    o = GamePlay.__new__(GamePlay)
    o.bb = BaseChessBoard(fen)
    o.red = o.bb.move_side is not ChessSide.BLACK
    return o


class PolicyHTTPEvalClient:
    """轮询多个 ``base_url``（各对应一评估子进程），阻塞 POST ``/eval``。"""

    def __init__(self, base_urls: list[str], *, timeout_s: float = 120.0) -> None:
        self._urls = [u.rstrip("/") for u in base_urls if u.strip()]
        if not self._urls:
            raise ValueError("PolicyHTTPEvalClient: 至少需要一个 base_url")
        self._timeout_s = float(timeout_s)
        self._rr = cycle(range(len(self._urls)))
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            i = next(self._rr)
        return self._urls[i]

    def eval_policy_value(self, gp: GamePlay) -> tuple[list[str], np.ndarray, float]:
        fen = gameplay_to_fen(gp)
        payload = json.dumps({"fen": fen}).encode("utf-8")
        url = self._next_url() + "/eval"
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"策略 HTTP 评估失败 {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"策略 HTTP 连接失败: {e}") from e
        data: dict[str, Any] = json.loads(raw.decode("utf-8"))
        if not data.get("ok", False):
            raise RuntimeError(f"策略 HTTP 返回错误: {data!r}")
        legals = list(data["legals"])
        priors = np.asarray(data["priors"], dtype=np.float64)
        v = float(data["v"])
        return legals, priors, v


def make_http_evaluator(client: PolicyHTTPEvalClient):
    return client.eval_policy_value


def wait_http_eval_ready(base_urls: list[str], *, timeout_s: float = 60.0, interval_s: float = 0.15) -> None:
    import time

    t0 = time.perf_counter()
    urls = [u.rstrip("/") for u in base_urls]
    pending = set(urls)
    last_err: str | None = None
    while time.perf_counter() - t0 < timeout_s and pending:
        for u in list(pending):
            try:
                req = urllib.request.Request(u + "/health", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as r:
                    if r.status == 200 and r.read(8):
                        pending.discard(u)
            except Exception as e:
                last_err = str(e)
        if not pending:
            return
        time.sleep(interval_s)
    raise RuntimeError(f"策略 HTTP 子进程在 {timeout_s:.0f}s 内未就绪: 仍无响应 {pending!r} 最后错误: {last_err}")


def spawn_mcts_http_eval_cluster(
    checkpoint: Path,
    n_workers: int,
    base_port: int,
    *,
    in_channels: int | None,
    host: str = "127.0.0.1",
    gpu: int = 0,
) -> tuple[list[subprocess.Popen], list[str]]:
    """启动 ``n_workers`` 个本机 HTTP 评估子进程（各占用 ``base_port+i``）。"""
    if n_workers <= 0:
        raise ValueError("spawn_mcts_http_eval_cluster: n_workers 须为正")
    procs: list[subprocess.Popen] = []
    urls: list[str] = []
    creationflags = 0
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW
    ck = str(checkpoint.resolve())
    for i in range(n_workers):
        port = int(base_port) + i
        cmd = [
            sys.executable,
            "-m",
            "my_elephant.training.policy_eval_worker",
            "--checkpoint",
            ck,
            "--host",
            host,
            "--port",
            str(port),
        ]
        if in_channels is not None:
            cmd += ["--in-channels", str(in_channels)]
        cmd += ["--gpu", str(int(gpu))]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        procs.append(p)
        urls.append(f"http://{host}:{port}")
    wait_http_eval_ready(urls)
    return procs, urls


def terminate_http_eval_cluster(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        except Exception:
            pass


def run_eval_http_server(
    host: str,
    port: int,
    model: SuccessorPolicy,
    device: torch.device,
    flist: dict[str, list[str]],
) -> None:
    """阻塞运行单线程 ``HTTPServer``（每子进程一个端口）。"""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/health", "/health/"):
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path not in ("/eval", "/eval/"):
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n > 0 else b"{}"
                obj = json.loads(raw.decode("utf-8"))
                fen = str(obj.get("fen", "")).strip()
                if not fen:
                    self._json_err(400, "missing fen")
                    return
                gp = gameplay_from_fen(fen)
                legals, priors, v = eval_policy_value_at_root(gp, model, device, flist)
                out = {
                    "ok": True,
                    "legals": legals,
                    "priors": [float(x) for x in priors.tolist()],
                    "v": float(v),
                }
                body = json.dumps(out).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._json_err(500, str(e))

        def _json_err(self, code: int, msg: str) -> None:
            body = json.dumps({"ok": False, "error": msg}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer((host, port), Handler)
    server.serve_forever()
