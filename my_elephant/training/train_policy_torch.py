"""
使用 PyTorch 训练「合法着法后继局面」打分策略网络：对 logits 做 CE，标签为棋谱着法下标。

依赖：pip install torch pandas（GPU 需对应 CUDA 版 torch）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from my_elephant.datasets import ProgressBar
from my_elephant.training.policy_data import make_policy_dataloader
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    accuracy_from_logits_masked,
    batched_successors_nhwc_to_torch,
    logits_as_red_preference,
    torch_load_checkpoint,
)


class ExpVal:
    """对 batch 标量做指数平滑。"""

    def __init__(self, exp_a: float = 0.97) -> None:
        self.val: float | None = None
        self.exp_a = exp_a

    def update(self, newval: float) -> None:
        if self.val is None:
            self.val = float(newval)
        else:
            self.val = self.exp_a * self.val + (1 - self.exp_a) * float(newval)

    def getval(self) -> float:
        assert self.val is not None
        return round(self.val, 2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练象棋策略网络 (PyTorch，合法着法 CE)")
    p.add_argument("--train-list", type=Path, default=Path("data/train_list.csv"))
    p.add_argument("--test-list", type=Path, default=Path("data/test_list.csv"))
    p.add_argument("--model-name", type=str, default="policy_resnet_torch")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每步前向为 B×K 个子局面，显存紧张时可减小",
    )
    p.add_argument("--num-res-layers", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0, help="GPU 编号；-1 表示 CPU")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--n-batch", type=int, default=10000)
    p.add_argument("--n-batch-test", type=int, default=300)
    p.add_argument(
        "--beginning-lr",
        "--lr",
        type=float,
        default=0.03,
        dest="beginning_lr",
        metavar="LR",
        help="SGD 学习率初值；按 --decay-epoch 分段乘以 0.1",
    )
    p.add_argument("--decay-epoch", type=int, default=10)
    p.add_argument("--log-dir", type=Path, default=Path("log"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从 checkpoint 恢复（如 models/<name>/last.pt 或 best.pt）",
    )
    p.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="输入通道数（默认含棋理附加层）",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 子进程数（在 worker 里解析 XML、枚举合法着法）；0 表示主进程加载",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="每个 worker 预取的 batch 数（仅 num_workers>0；略增内存换吞吐）",
    )
    return p.parse_args()


def _device(args: argparse.Namespace) -> torch.device:
    if args.gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu}")


def main() -> None:
    args = _parse_args()
    device = _device(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.model_dir / args.model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch_load_checkpoint(args.resume, device) if args.resume is not None else None
    if ckpt is not None:
        in_ch = (
            args.in_channels
            if args.in_channels is not None
            else int(ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS)))
        )
    else:
        in_ch = args.in_channels if args.in_channels is not None else POLICY_SELECT_IN_CHANNELS

    model = SuccessorPolicy(num_res_layers=args.num_res_layers, in_channels=in_ch).to(device)
    optimizer = SGD(model.parameters(), lr=args.beginning_lr, momentum=0.9)

    epoch_begin = 0
    global_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch_begin = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))

    best_val_loss = float("inf")
    if ckpt is not None:
        b = ckpt.get("best_val_loss")
        if isinstance(b, float):
            best_val_loss = b
        elif isinstance(b, int) and not isinstance(b, bool):
            best_val_loss = float(b)

    writer = SummaryWriter(log_dir=str(args.log_dir / f"{args.model_name}_torch"))

    pin_mem = device.type == "cuda"
    train_loader = make_policy_dataloader(
        args.train_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_policy_dataloader(
        args.test_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        prefetch_factor=args.prefetch_factor,
    )
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    for epoch in range(epoch_begin, args.n_epochs):
        model.train()
        prev_lr = float(optimizer.param_groups[0]["lr"])
        batch_lr = args.beginning_lr * 10 ** -(epoch // args.decay_epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = batch_lr
        if batch_lr < prev_lr - 1e-12:
            print(f"\n[LR] epoch {epoch}: {prev_lr:g} -> {batch_lr:g}\n")

        expacc = ExpVal()
        exploss = ExpVal()

        pb = ProgressBar(worksum=args.n_batch * args.batch_size, info=f"epoch {epoch} train")
        pb.startjob()
        for batch_i in range(args.n_batch):
            try:
                batch_x, batch_mask, batch_y, batch_red = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_mask, batch_y, batch_red = next(train_iter)
            x = batched_successors_nhwc_to_torch(
                batch_x, device, pin_memory=pin_mem, non_blocking=pin_mem
            )
            mask = torch.as_tensor(batch_mask, dtype=torch.bool)
            target = torch.as_tensor(batch_y, dtype=torch.long)
            red_b = torch.as_tensor(batch_red, dtype=torch.bool)
            if pin_mem:
                mask = mask.pin_memory()
                target = target.pin_memory()
                red_b = red_b.pin_memory()
            mask = mask.to(device, non_blocking=pin_mem)
            target = target.to(device, non_blocking=pin_mem)
            red_b = red_b.to(device, non_blocking=pin_mem)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            logits_r = logits_as_red_preference(logits, red_b)
            logits_r = logits_r.masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits_r, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits_masked(logits_r, target, mask)

            global_step += 1
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/acc_move_choice", float(acc.item()), global_step)
            writer.add_scalar("train/lr", batch_lr, global_step)

            expacc.update(float(acc.item()) * 100)
            exploss.update(float(loss.item()))
            pb.info = (
                f"EPOCH {epoch} STEP {batch_i} LR {batch_lr} "
                f"ACC {expacc.getval()}% (move) LOSS {exploss.getval()} "
            )
            pb.complete(args.batch_size)
        print()

        model.eval()
        accs: list[float] = []
        losses: list[float] = []
        pb = ProgressBar(worksum=args.n_batch_test * args.batch_size, info=f"validating epoch {epoch}")
        pb.startjob()
        with torch.no_grad():
            for _ in range(args.n_batch_test):
                try:
                    batch_x, batch_mask, batch_y, batch_red = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    batch_x, batch_mask, batch_y, batch_red = next(test_iter)
                x = batched_successors_nhwc_to_torch(
                    batch_x, device, pin_memory=pin_mem, non_blocking=pin_mem
                )
                mask = torch.as_tensor(batch_mask, dtype=torch.bool)
                target = torch.as_tensor(batch_y, dtype=torch.long)
                red_b = torch.as_tensor(batch_red, dtype=torch.bool)
                if pin_mem:
                    mask = mask.pin_memory()
                    target = target.pin_memory()
                    red_b = red_b.pin_memory()
                mask = mask.to(device, non_blocking=pin_mem)
                target = target.to(device, non_blocking=pin_mem)
                red_b = red_b.to(device, non_blocking=pin_mem)
                logits = model(x)
                logits_r = logits_as_red_preference(logits, red_b)
                logits_m = logits_r.masked_fill(~mask, -1e9)
                loss = F.cross_entropy(logits_m, target)
                acc = accuracy_from_logits_masked(logits_m, target, mask)
                accs.append(float(acc.item()))
                losses.append(float(loss.item()))
                pb.complete(args.batch_size)
        print(
            "TEST ACC(move)%",
            100.0 * np.average(accs),
            "LOSS",
            np.average(losses),
        )
        writer.add_scalar("val/loss", float(np.average(losses)), epoch)
        writer.add_scalar("val/acc_move_choice", float(np.average(accs)), epoch)
        print()

        val_loss = float(np.average(losses))
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "num_res_layers": args.num_res_layers,
            "in_channels": model.in_channels,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if improved:
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"  -> saved best.pt (val_loss={val_loss:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
