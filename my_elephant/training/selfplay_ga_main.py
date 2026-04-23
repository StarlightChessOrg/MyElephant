"""
并行 MCTS 自对弈 + 极简遗传算法：多进程产棋谱 → 按个体缓冲算适应度（策略 NLL）→
选择 / 交叉（状态字典插值）/ 变异 → 在合并棋谱上做少量 SGD 微调。

入口::

    python -m my_elephant.training.selfplay_ga_main --init-checkpoint path/to/best.pt --work-dir runs/ga0

依赖与 ``train_policy_torch`` 相同；默认 CPU；Windows 下请直接运行本模块（spawn 安全）。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

from my_elephant.chess.rationale import POLICY_GRID_NUMEL, VALUE_LABEL_IGNORE
from my_elephant.training.play_model_loader import load_successor_policy_for_play
from my_elephant.training.policy_data import collate_twohead_policy_value_batch
from my_elephant.training.policy_torch import (
    batched_current_nhwc_to_torch,
    torch_load_checkpoint,
)
from my_elephant.training.selfplay_sample import play_one_selfplay_mcts_game


def _blend_state_dicts(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in a:
        ta, tb = a[k], b[k]
        if ta.shape == tb.shape and ta.dtype.is_floating_point and tb.dtype.is_floating_point:
            out[k] = ((1.0 - alpha) * ta + alpha * tb).cpu().clone()
        else:
            out[k] = ta.clone()
    return out


def _mutate_state_dict(sd: dict[str, torch.Tensor], sigma: float, rng: random.Random) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, t in sd.items():
        if t.dtype.is_floating_point:
            noise = torch.randn_like(t) * sigma
            out[k] = (t + noise).cpu().clone()
        else:
            out[k] = t.clone()
    return out


def _policy_nll_on_buffer(
    model: torch.nn.Module,
    buffer: list[tuple],
    device: torch.device,
    *,
    max_batches: int = 24,
    batch_size: int = 32,
) -> float:
    """越小越好：在缓冲上平均策略 CE（无梯度）。"""
    if not buffer:
        return 1e9
    rng = random.Random(0)
    idx = list(range(len(buffer)))
    rng.shuffle(idx)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for bi in range(min(max_batches, math.ceil(len(buffer) / batch_size))):
            chunk = [buffer[i] for i in idx[bi * batch_size : (bi + 1) * batch_size]]
            if not chunk:
                break
            cur, msrc, mdst, ys, yd, _red, yv = collate_twohead_policy_value_batch(chunk)
            x_cur = batched_current_nhwc_to_torch(torch.from_numpy(cur), device)
            msrc_t = torch.as_tensor(msrc, dtype=torch.bool, device=device)
            mdst_t = torch.as_tensor(mdst, dtype=torch.bool, device=device)
            tgt_s = torch.as_tensor(ys, dtype=torch.long, device=device)
            tgt_d = torch.as_tensor(yd, dtype=torch.long, device=device)
            src_oh = F.one_hot(tgt_s, num_classes=POLICY_GRID_NUMEL).float()
            logits_s, logits_d, _lv = model(x_cur, src_oh)
            loss_s = F.cross_entropy(logits_s.masked_fill(~msrc_t, -1e9), tgt_s)
            loss_d = F.cross_entropy(logits_d.masked_fill(~mdst_t, -1e9), tgt_d)
            tot += float((loss_s + loss_d).item())
            n += 1
    return tot / max(n, 1)


def _fine_tune_checkpoint(
    ckpt_in: Path,
    buffer: list[tuple],
    ckpt_out: Path,
    *,
    device: torch.device,
    in_channels: int | None,
    steps: int,
    lr: float,
    batch_size: int,
    value_loss_weight: float,
) -> None:
    if not buffer:
        torch.save(torch.load(ckpt_in, map_location="cpu", weights_only=False), ckpt_out)
        return
    model, flist = load_successor_policy_for_play(ckpt_in, device, in_channels=in_channels)
    model.train()
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    rng = random.Random(0)
    idx = list(range(len(buffer)))
    for step in range(steps):
        rng.shuffle(idx)
        chunk = [buffer[i] for i in idx[:batch_size]]
        if len(chunk) < 2:
            continue
        cur, msrc, mdst, ys, yd, _red, yv = collate_twohead_policy_value_batch(chunk)
        x_cur = batched_current_nhwc_to_torch(torch.from_numpy(cur), device)
        msrc_t = torch.as_tensor(msrc, dtype=torch.bool, device=device)
        mdst_t = torch.as_tensor(mdst, dtype=torch.bool, device=device)
        tgt_s = torch.as_tensor(ys, dtype=torch.long, device=device)
        tgt_d = torch.as_tensor(yd, dtype=torch.long, device=device)
        target_v = torch.as_tensor(yv, dtype=torch.long, device=device)
        src_oh = F.one_hot(tgt_s, num_classes=POLICY_GRID_NUMEL).float()
        opt.zero_grad(set_to_none=True)
        logits_s, logits_d, logits_v = model(x_cur, src_oh)
        loss_s = F.cross_entropy(logits_s.masked_fill(~msrc_t, -1e9), tgt_s)
        loss_d = F.cross_entropy(logits_d.masked_fill(~mdst_t, -1e9), tgt_d)
        labeled_v = target_v != VALUE_LABEL_IGNORE
        if labeled_v.any():
            loss_v = F.cross_entropy(logits_v, target_v, ignore_index=VALUE_LABEL_IGNORE)
        else:
            loss_v = logits_v.sum() * 0.0
        loss = loss_s + loss_d + value_loss_weight * loss_v
        loss.backward()
        opt.step()

    meta = torch_load_checkpoint(ckpt_in, device)
    meta["model"] = model.state_dict()
    torch.save(meta, ckpt_out)


def _spawn_population(init_ckpt: Path, work_dir: Path, pop_size: int, noise: float) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    base = torch_load_checkpoint(init_ckpt, torch.device("cpu"), weights_only=False)
    sd0 = base["model"]
    paths: list[Path] = []
    rng = random.Random(42)
    for i in range(pop_size):
        sd = {k: v.clone() for k, v in sd0.items()}
        if i > 0:
            sd = _mutate_state_dict(sd, noise, rng)
        p = work_dir / f"pop_init_{i}.pt"
        payload = {**{k: v for k, v in base.items() if k != "model"}, "model": sd}
        torch.save(payload, p)
        paths.append(p)
    return paths


def _mp_play_games(args: tuple) -> tuple[int, list]:
    (
        ckpt_str,
        org_id,
        n_games,
        mcts_sims,
        mcts_max_seconds,
        mcts_workers,
        in_ch,
    ) = args
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = torch.device("cpu")
    model, flist = load_successor_policy_for_play(Path(ckpt_str), device, in_channels=in_ch)
    buf: list = []
    for _ in range(int(n_games)):
        buf.extend(
            play_one_selfplay_mcts_game(
                model,
                device,
                flist,
                mcts_sims=mcts_sims,
                mcts_max_seconds=mcts_max_seconds,
                mcts_workers=mcts_workers,
                thread_pool=None,
            )
        )
    return int(org_id), buf


def main() -> None:
    p = argparse.ArgumentParser(description="并行 MCTS 自对弈 + 遗传算法微调")
    p.add_argument("--init-checkpoint", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("runs/selfplay_ga"))
    p.add_argument("--pop-size", type=int, default=4)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--games-per-org", type=int, default=2, help="每个个体每代并行自对弈局数（进程池任务数=种群×本值）")
    p.add_argument("--parallel-workers", type=int, default=None, help="进程池大小，默认同 CPU 数")
    p.add_argument("--mcts-sims", type=int, default=96)
    p.add_argument("--mcts-max-seconds", type=float, default=6.0)
    p.add_argument("--mcts-play-workers", type=int, default=1, help="单局 MCTS 线程数（子进程内）")
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--init-noise", type=float, default=0.02, help="初始种群除 0 号外对权重的加性高斯噪声标准差")
    p.add_argument("--cross-alpha", type=float, default=0.5, help="交叉：两父状态字典插值系数")
    p.add_argument("--mutate-sigma", type=float, default=0.015, help="子代变异噪声")
    p.add_argument("--train-steps", type=int, default=80, help="每代末在合并棋谱上对每个存活个体 SGD 步数")
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--train-lr", type=float, default=0.02)
    p.add_argument("--value-loss-weight", type=float, default=0.25)
    args = p.parse_args()

    work_dir = args.work_dir.resolve()
    pop_dir = work_dir / "population"
    pop_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    pop_paths = _spawn_population(args.init_checkpoint.resolve(), pop_dir, args.pop_size, args.init_noise)
    pool_n = args.parallel_workers or (os.cpu_count() or 2)
    pool_n = max(1, min(pool_n, args.pop_size * max(1, args.games_per_org)))

    for gen in range(args.generations):
        print(f"\n=== generation {gen} ===")
        tasks = []
        for oid, ck in enumerate(pop_paths):
            for _ in range(args.games_per_org):
                tasks.append(
                    (
                        str(ck),
                        oid,
                        1,
                        args.mcts_sims,
                        args.mcts_max_seconds,
                        args.mcts_play_workers,
                        args.in_channels,
                    )
                )
        per_org: list[list] = [[] for _ in range(len(pop_paths))]
        with ProcessPoolExecutor(max_workers=pool_n) as ex:
            futs = [ex.submit(_mp_play_games, t) for t in tasks]
            for fu in as_completed(futs):
                oid, buf = fu.result()
                per_org[oid].extend(buf)
        merged: list = []
        for b in per_org:
            merged.extend(b)
        print(f"  samples total={len(merged)} (per-org: {[len(x) for x in per_org]})")

        fitness: list[tuple[float, int]] = []
        for oid, ck in enumerate(pop_paths):
            model, _fl = load_successor_policy_for_play(ck, device, in_channels=args.in_channels)
            nll = _policy_nll_on_buffer(model, per_org[oid], device)
            fitness.append((nll, oid))
            print(f"  org {oid} nll~{nll:.4f} (lower better)")

        fitness.sort(key=lambda x: x[0])
        elite_n = max(2, args.pop_size // 2)
        elites = [oid for _, oid in fitness[:elite_n]]
        print(f"  elites (oid): {elites}")

        new_paths: list[Path] = []
        rng = random.Random(gen + 11)
        sd_elite = [
            torch_load_checkpoint(pop_paths[i], device, weights_only=False)["model"] for i in elites
        ]
        for j in range(args.pop_size):
            if j < len(elites):
                p_out = pop_dir / f"gen{gen}_keep_{j}.pt"
                torch.save(torch.load(pop_paths[elites[j]], map_location="cpu", weights_only=False), p_out)
                new_paths.append(p_out)
                continue
            ia, ib = rng.choice(elites), rng.choice(elites)
            alpha = rng.random() * float(args.cross_alpha) + (1.0 - float(args.cross_alpha)) * 0.5
            blended = _blend_state_dicts(sd_elite[elites.index(ia)], sd_elite[elites.index(ib)], alpha)
            blended = _mutate_state_dict(blended, float(args.mutate_sigma), rng)
            meta = torch_load_checkpoint(pop_paths[ia], device, weights_only=False)
            meta["model"] = blended
            p_out = pop_dir / f"gen{gen}_child_{j}.pt"
            torch.save(meta, p_out)
            new_paths.append(p_out)

        ft_path = work_dir / f"gen{gen}_merged_finetune.pt"
        _fine_tune_checkpoint(
            new_paths[0],
            merged,
            ft_path,
            device=device,
            in_channels=args.in_channels,
            steps=int(args.train_steps),
            lr=float(args.train_lr),
            batch_size=int(args.train_batch),
            value_loss_weight=float(args.value_loss_weight),
        )
        meta_best = torch_load_checkpoint(ft_path, device, weights_only=False)
        for i, pp in enumerate(new_paths):
            m = dict(meta_best)
            m["model"] = torch_load_checkpoint(pp, device, weights_only=False)["model"]
            torch.save(m, pp)

        pop_paths = new_paths
        print(f"  generation {gen} done; population dir: {pop_dir}")

    final_p = work_dir / "final_best.pt"
    torch.save(torch.load(pop_paths[0], map_location="cpu", weights_only=False), final_p)
    print(f"\n完成。推荐检查点（精英 0 号末代）: {pop_paths[0]}\n合并微调副本: {final_p}")


if __name__ == "__main__":
    if sys.platform == "win32":
        multiprocessing.freeze_support()  # type: ignore[name-defined]
    main()
