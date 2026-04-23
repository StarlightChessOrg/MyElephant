"""
使用 PyTorch 训练「合法着法后继局面」打分策略网络（着法 CE）
与红方胜/和/负三分类价值头（``Head/RecordResult`` CE，与 ``cchess.reader_xqf`` 编码一致）。

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
from my_elephant.training.policy_data import (
    build_policy_train_val_loaders,
    default_num_workers,
    discover_cbf_files,
    split_paths_train_test,
)
from my_elephant.chess.rationale import POLICY_SELECT_IN_CHANNELS, VALUE_LABEL_IGNORE
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    accuracy_from_logits_masked,
    batched_current_nhwc_to_torch,
    batched_successors_nhwc_to_torch,
    logits_as_red_preference,
    torch_load_checkpoint,
    value_accuracy_ignore,
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
    p = argparse.ArgumentParser(description="训练象棋策略+价值网络 (PyTorch，着法 CE + 红方结果 CE)")
    p.add_argument(
        "--cbf-root",
        type=Path,
        default=None,
        help="若指定则从该目录搜集 .cbf（默认递归子目录），并按 --train-ratio 划分 train/val，不再读取 --train-list/--test-list",
    )
    p.add_argument(
        "--cbf-shallow",
        action="store_true",
        help="与 --cbf-root 配合：只搜索根目录一层，不递归子文件夹",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="仅在使用 --cbf-root 时生效：训练集文件占比 (0,1)",
    )
    p.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="仅在使用 --cbf-root 时生效：train/test 划分的随机种子",
    )
    p.add_argument("--train-list", type=Path, default=Path("data/train_list.csv"))
    p.add_argument("--test-list", type=Path, default=Path("data/test_list.csv"))
    p.add_argument("--model-name", type=str, default="policy_resnet_torch")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每步前向为 B×K 个子局面，显存紧张时可减小",
    )
    p.add_argument(
        "--num-res-layers",
        type=int,
        default=4,
        help="残差块层数（默认较小以便 MCTS 多次前向）",
    )
    p.add_argument(
        "--filters",
        type=int,
        default=64,
        help="卷积宽度（默认约数 MB 级权重，利于 MCTS；旧 checkpoint 请与训练时一致）",
    )
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
        help="从指定 checkpoint 恢复（权重、优化器、epoch、global_step、best_val_loss）",
    )
    p.add_argument(
        "--continue",
        action="store_true",
        dest="continue_train",
        help="续训：自动加载 models/<--model-name>/last.pt（存在则恢复，不存在则从头训；若同时给 --resume 则只用 --resume）",
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
        default=None,
        help="DataLoader 子进程数（并行解析棋谱）；默认 min(8, CPU 核数)；0 表示主进程加载",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="每个 worker 预取的 batch 数（仅 num_workers>0；略增内存换吞吐）",
    )
    p.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.5,
        help="红方胜/和/负价值头交叉熵相对策略 CE 的权重（无 RecordResult 标签的样本自动忽略）",
    )
    return p.parse_args()


def _device(args: argparse.Namespace) -> torch.device:
    if args.gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{args.gpu}")


def _resolve_resume_path(args: argparse.Namespace) -> Path | None:
    """``--resume`` 优先；否则 ``--continue`` 时尝试 ``model_dir/model_name/last.pt``。"""
    if args.resume is not None:
        if getattr(args, "continue_train", False):
            print("[续训] 已指定 --resume，忽略 --continue")
        return args.resume
    if getattr(args, "continue_train", False):
        last_pt = args.model_dir / args.model_name / "last.pt"
        if last_pt.is_file():
            print(f"[续训] 自动加载 {last_pt.resolve()}")
            return last_pt
        print(f"[续训] 未找到 {last_pt.resolve()}，将从头训练")
    return None


def main() -> None:
    args = _parse_args()
    if args.num_workers is None:
        args.num_workers = default_num_workers()
    device = _device(args)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.model_dir / args.model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = _resolve_resume_path(args)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_path.resolve()}")
    # 先在 CPU 上读权重，再在构建 DataLoader（多进程 fork/spawn）之后再 .to(cuda)，减轻 fork 后 CUDA 上下文问题
    ckpt = (
        torch_load_checkpoint(resume_path, torch.device("cpu"))
        if resume_path is not None
        else None
    )
    if ckpt is not None:
        in_ch = (
            args.in_channels
            if args.in_channels is not None
            else int(ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS)))
        )
        sd = ckpt.get("model", {})
        if isinstance(sd, dict) and "stem_conv.weight" in sd:
            inferred_f = int(sd["stem_conv.weight"].shape[0])
            if ckpt.get("filters") is not None:
                args.filters = int(ckpt["filters"])
            else:
                args.filters = inferred_f
        if ckpt.get("num_res_layers") is not None:
            args.num_res_layers = int(ckpt["num_res_layers"])
    else:
        in_ch = args.in_channels if args.in_channels is not None else POLICY_SELECT_IN_CHANNELS

    if args.cbf_root is not None:
        all_cbf = discover_cbf_files(args.cbf_root, recursive=not args.cbf_shallow)
        train_sources, test_sources = split_paths_train_test(
            all_cbf, args.train_ratio, seed=args.data_seed
        )
        print(
            f"[data] --cbf-root {args.cbf_root.resolve()}: "
            f"{len(all_cbf)} .cbf -> train {len(train_sources)} / test {len(test_sources)}"
        )
    else:
        train_sources, test_sources = args.train_list, args.test_list

    pin_mem = device.type == "cuda"
    pin_dev = str(device) if pin_mem else None
    train_loader, test_loader = build_policy_train_val_loaders(
        train_sources,
        test_sources,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_mem,
        pin_memory_device=pin_dev,
    )
    if args.num_workers > 0:
        print(
            f"[data] DataLoader: num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}, "
            f"pin_memory={pin_mem}, train_drop_last=True"
        )

    model = SuccessorPolicy(
        num_res_layers=args.num_res_layers, in_channels=in_ch, filters=args.filters
    ).to(device)
    optimizer = SGD(model.parameters(), lr=args.beginning_lr, momentum=0.9)

    epoch_begin = 0
    global_step = 0
    if ckpt is not None:
        load_ret = model.load_state_dict(ckpt["model"], strict=False)
        if load_ret.missing_keys:
            mk = sorted(load_ret.missing_keys)
            tail = "..." if len(mk) > 12 else ""
            print(f"[resume] 未从 checkpoint 加载的键（将用随机初始化）: {mk[:12]}{tail}")
        if load_ret.unexpected_keys:
            print(f"[resume] checkpoint 中多余键: {load_ret.unexpected_keys}")
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
        expacc_v = ExpVal()
        exploss_v = ExpVal()

        pb = ProgressBar(worksum=args.n_batch * args.batch_size, info=f"epoch {epoch} train")
        pb.startjob()
        for batch_i in range(args.n_batch):
            try:
                batch_x, batch_mask, batch_y, batch_red, batch_cur, batch_yv = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_x, batch_mask, batch_y, batch_red, batch_cur, batch_yv = next(train_iter)
            x = batched_successors_nhwc_to_torch(
                batch_x, device, pin_memory=pin_mem, non_blocking=pin_mem
            )
            x_cur = batched_current_nhwc_to_torch(
                batch_cur, device, pin_memory=pin_mem, non_blocking=pin_mem
            )
            mask = torch.as_tensor(batch_mask, dtype=torch.bool)
            target = torch.as_tensor(batch_y, dtype=torch.long)
            red_b = torch.as_tensor(batch_red, dtype=torch.bool)
            target_v = torch.as_tensor(batch_yv, dtype=torch.long)
            if pin_mem:
                mask = mask.pin_memory()
                target = target.pin_memory()
                red_b = red_b.pin_memory()
                target_v = target_v.pin_memory()
            mask = mask.to(device, non_blocking=pin_mem)
            target = target.to(device, non_blocking=pin_mem)
            red_b = red_b.to(device, non_blocking=pin_mem)
            target_v = target_v.to(device, non_blocking=pin_mem)

            optimizer.zero_grad(set_to_none=True)
            logits_p, logits_v = model(x, x_cur)
            logits_r = logits_as_red_preference(logits_p, red_b)
            logits_r = logits_r.masked_fill(~mask, -1e9)
            loss_p = F.cross_entropy(logits_r, target)
            labeled_v = target_v != VALUE_LABEL_IGNORE
            if labeled_v.any():
                loss_v = F.cross_entropy(
                    logits_v, target_v, ignore_index=VALUE_LABEL_IGNORE
                )
            else:
                loss_v = (logits_v * 0).sum()
            loss = loss_p + args.value_loss_weight * loss_v
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_from_logits_masked(logits_r, target, mask)
                acc_v = value_accuracy_ignore(logits_v, target_v)

            global_step += 1
            writer.add_scalar("train/loss_policy", float(loss_p.item()), global_step)
            writer.add_scalar("train/loss_value", float(loss_v.item()), global_step)
            writer.add_scalar("train/loss_total", float(loss.item()), global_step)
            writer.add_scalar("train/acc_move_choice", float(acc.item()), global_step)
            writer.add_scalar("train/acc_red_outcome", float(acc_v.item()), global_step)
            writer.add_scalar("train/lr", batch_lr, global_step)

            expacc.update(float(acc.item()) * 100)
            exploss.update(float(loss.item()))
            expacc_v.update(float(acc_v.item()) * 100)
            exploss_v.update(float(loss_v.item()))
            pb.info = (
                f"EPOCH {epoch} STEP {batch_i} LR {batch_lr} "
                f"ACCmv {expacc.getval()}% ACCout {expacc_v.getval()}% "
                f"LOSS {exploss.getval()} (v {exploss_v.getval()}) "
            )
            pb.complete(args.batch_size)
        print()

        model.eval()
        accs: list[float] = []
        accs_v: list[float] = []
        losses: list[float] = []
        losses_p: list[float] = []
        losses_v: list[float] = []
        pb = ProgressBar(worksum=args.n_batch_test * args.batch_size, info=f"validating epoch {epoch}")
        pb.startjob()
        with torch.no_grad():
            for _ in range(args.n_batch_test):
                try:
                    batch_x, batch_mask, batch_y, batch_red, batch_cur, batch_yv = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    batch_x, batch_mask, batch_y, batch_red, batch_cur, batch_yv = next(test_iter)
                x = batched_successors_nhwc_to_torch(
                    batch_x, device, pin_memory=pin_mem, non_blocking=pin_mem
                )
                x_cur = batched_current_nhwc_to_torch(
                    batch_cur, device, pin_memory=pin_mem, non_blocking=pin_mem
                )
                mask = torch.as_tensor(batch_mask, dtype=torch.bool)
                target = torch.as_tensor(batch_y, dtype=torch.long)
                red_b = torch.as_tensor(batch_red, dtype=torch.bool)
                target_v = torch.as_tensor(batch_yv, dtype=torch.long)
                if pin_mem:
                    mask = mask.pin_memory()
                    target = target.pin_memory()
                    red_b = red_b.pin_memory()
                    target_v = target_v.pin_memory()
                mask = mask.to(device, non_blocking=pin_mem)
                target = target.to(device, non_blocking=pin_mem)
                red_b = red_b.to(device, non_blocking=pin_mem)
                target_v = target_v.to(device, non_blocking=pin_mem)
                logits_p, logits_v = model(x, x_cur)
                logits_r = logits_as_red_preference(logits_p, red_b)
                logits_m = logits_r.masked_fill(~mask, -1e9)
                loss_p = F.cross_entropy(logits_m, target)
                labeled_v = target_v != VALUE_LABEL_IGNORE
                if labeled_v.any():
                    loss_v = F.cross_entropy(
                        logits_v, target_v, ignore_index=VALUE_LABEL_IGNORE
                    )
                else:
                    loss_v = (logits_v * 0).sum()
                loss = loss_p + args.value_loss_weight * loss_v
                acc = accuracy_from_logits_masked(logits_m, target, mask)
                acc_v = value_accuracy_ignore(logits_v, target_v)
                accs.append(float(acc.item()))
                accs_v.append(float(acc_v.item()))
                losses.append(float(loss.item()))
                losses_p.append(float(loss_p.item()))
                losses_v.append(float(loss_v.item()))
                pb.complete(args.batch_size)
        print(
            "TEST ACC(move)%",
            100.0 * np.average(accs),
            "ACC(red-out)%",
            100.0 * np.average(accs_v),
            "LOSS total",
            np.average(losses),
            "LOSS pol",
            np.average(losses_p),
            "LOSS val",
            np.average(losses_v),
        )
        writer.add_scalar("val/loss_total", float(np.average(losses)), epoch)
        writer.add_scalar("val/loss_policy", float(np.average(losses_p)), epoch)
        writer.add_scalar("val/loss_value", float(np.average(losses_v)), epoch)
        writer.add_scalar("val/acc_move_choice", float(np.average(accs)), epoch)
        writer.add_scalar("val/acc_red_outcome", float(np.average(accs_v)), epoch)
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
            "filters": model.filters,
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
