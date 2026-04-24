"""
使用 PyTorch 训练两阶段策略（起点格 + 落点格 CE）与**行棋方**胜/和/负价值头
（``Head/RecordResult`` CE，与 ``cchess.reader_xqf`` 编码一致）。主干为 **ResNet 卷积塔**（与 ``StarlightChessOrg/MyElephant`` 提交 ``91e27b25`` 中塔结构一致）+ 当前两阶段策略头与三分类价值头。

优化器为 **RAdam**（自适应学习率 + 整流预热）；依赖：pip install torch pandas（GPU 需对应 CUDA 版 torch；需 PyTorch≥1.12 含 ``RAdam``）。
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RAdam
from torch.utils.tensorboard import SummaryWriter

from my_elephant.datasets import ProgressBar
from my_elephant.training.policy_data import (
    build_policy_train_val_loaders,
    default_num_workers,
    discover_cbf_files,
    split_paths_train_test,
)
from my_elephant.chess.rationale import POLICY_GRID_NUMEL, POLICY_SELECT_IN_CHANNELS, VALUE_LABEL_IGNORE
from my_elephant.training.policy_torch import (
    SuccessorPolicy,
    accuracy_from_logits_masked,
    batched_current_nhwc_to_torch,
    count_resnet_blocks_in_state,
    joint_move_accuracy,
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


def _build_radam_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> RAdam:
    """构造 RAdam；在支持的 PyTorch 版本上对 weight_decay>0 启用解耦衰减（AdamW 风格）。"""
    wd = float(args.weight_decay)
    kw: dict = {
        "lr": float(args.beginning_lr),
        "betas": (float(args.beta1), float(args.beta2)),
        "eps": float(args.eps),
        "weight_decay": wd,
    }
    if wd > 0 and "decoupled_weight_decay" in inspect.signature(RAdam.__init__).parameters:
        kw["decoupled_weight_decay"] = True
    return RAdam(model.parameters(), **kw)


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
        help="每步 batch；显存紧张时可减小",
    )
    p.add_argument(
        "--num-res-layers",
        type=int,
        default=10,
        help="ResNet 残差块数（与 commit 91e27b25 默认一致；``--resume`` 时可从权重推断）",
    )
    p.add_argument(
        "--filters",
        type=int,
        default=256,
        help="卷积宽度（stem 与各 ResBlock 通道；旧 checkpoint 请与训练时一致）",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU 编号；-1 表示 CPU")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--n-batch", type=int, default=10000)
    p.add_argument("--n-batch-test", type=int, default=300)
    p.add_argument(
        "--beginning-lr",
        "--lr",
        type=float,
        default=1e-3,
        dest="beginning_lr",
        metavar="LR",
        help="RAdam 初始学习率；每经过 --decay-epoch 个 epoch 乘以 --lr-decay-factor",
    )
    p.add_argument(
        "--decay-epoch",
        type=int,
        default=15,
        help="学习率分段衰减的周期（epoch）；须 >=1",
    )
    p.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="每个衰减段将当前 lr 乘以此系数（0<系数≤1；1 表示不衰减）",
    )
    p.add_argument("--beta1", type=float, default=0.9, help="RAdam β1")
    p.add_argument("--beta2", type=float, default=0.999, help="RAdam β2")
    p.add_argument("--eps", type=float, default=1e-8, help="RAdam 数值稳定项 ε")
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
        help="行棋方胜/和/负价值头交叉熵相对策略 CE 的权重（无 RecordResult 标签的样本自动忽略）",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="RAdam 权重衰减；默认与先前 SGD 量级一致；可试 1e-3～1e-2（若 PyTorch 支持解耦式则自动启用）",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="验证 total loss 连续若干 epoch 未刷新 best 则停止；0 表示关闭（仍保存 best.pt）",
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
    if not (0.0 < float(args.lr_decay_factor) <= 1.0):
        raise SystemExit("--lr-decay-factor 须在 (0,1] 内")
    if int(args.decay_epoch) < 1:
        raise SystemExit("--decay-epoch 须 >= 1")
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
        sd = ckpt.get("model", {})
        if not isinstance(sd, dict):
            sd = {}
        if any(k.startswith("hybrid_trunk.") for k in sd) or str(ckpt.get("backbone", "")).lower() == "hybrid":
            raise ValueError(
                "checkpoint 为已移除的 hybrid 主干，无法加载；请换用 ResNet 两阶段权重或回退到仍含 hybrid 的仓库版本。"
            )
        if any(k.startswith("xfm_trunk.") for k in sd) or str(ckpt.get("backbone", "")).lower() == "transformer":
            raise ValueError(
                "checkpoint 为 transformer 主干，本版本已改为仅 ResNet 塔（见 commit 91e27b25 塔结构 + 两阶段头）；请换用 resnet 权重或使用旧分支。"
            )
        in_ch = (
            args.in_channels
            if args.in_channels is not None
            else int(ckpt.get("in_channels", ckpt.get("select_in_channels", POLICY_SELECT_IN_CHANNELS)))
        )
        if isinstance(sd, dict) and "stem_conv.weight" in sd:
            inferred_f = int(sd["stem_conv.weight"].shape[0])
            if ckpt.get("filters") is not None:
                args.filters = int(ckpt["filters"])
            else:
                args.filters = inferred_f
        if ckpt.get("num_res_layers") is not None:
            args.num_res_layers = int(ckpt["num_res_layers"])
        elif isinstance(sd, dict):
            nlay = count_resnet_blocks_in_state(sd)
            if nlay > 0:
                args.num_res_layers = nlay
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
        num_res_layers=args.num_res_layers,
        in_channels=in_ch,
        filters=args.filters,
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(
        f"[model] backbone=resnet filters={args.filters} res_blocks={args.num_res_layers} params={nparam:,}"
    )
    optimizer = _build_radam_optimizer(model, args)
    print(
        f"[optim] RAdam lr={args.beginning_lr:g} betas=({args.beta1},{args.beta2}) eps={args.eps:g} "
        f"weight_decay={args.weight_decay:g} decay_every={args.decay_epoch}epoch×{args.lr_decay_factor:g}"
    )

    epoch_begin = 0
    global_step = 0
    optimizer_state_loaded = False
    if ckpt is not None:
        load_ret = model.load_state_dict(ckpt["model"], strict=False)
        if load_ret.missing_keys:
            mk = sorted(load_ret.missing_keys)
            tail = "..." if len(mk) > 12 else ""
            print(f"[resume] 未从 checkpoint 加载的键（将用随机初始化）: {mk[:12]}{tail}")
        if load_ret.unexpected_keys:
            print(f"[resume] checkpoint 中多余键: {load_ret.unexpected_keys}")
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                optimizer_state_loaded = True
            except (ValueError, RuntimeError) as e:
                print(f"[resume] 优化器状态与当前 RAdam 不兼容，已跳过加载（将重新累积动量等）: {e}")
        epoch_begin = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
    if epoch_begin > 0 and not optimizer_state_loaded:
        lr_step = epoch_begin // int(args.decay_epoch)
        sync_lr = float(args.beginning_lr) * (float(args.lr_decay_factor) ** lr_step)
        for pg in optimizer.param_groups:
            pg["lr"] = sync_lr
        print(f"[resume] 优化器 lr 已按 epoch {epoch_begin} 对齐为 {sync_lr:g}")

    best_val_loss = float("inf")
    if ckpt is not None:
        b = ckpt.get("best_val_loss")
        if isinstance(b, float):
            best_val_loss = b
        elif isinstance(b, int) and not isinstance(b, bool):
            best_val_loss = float(b)

    epochs_no_improve = 0
    if ckpt is not None:
        ei = ckpt.get("epochs_no_improve")
        if isinstance(ei, int) and not isinstance(ei, bool):
            epochs_no_improve = int(ei)

    writer = SummaryWriter(log_dir=str(args.log_dir / f"{args.model_name}_torch"))

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    for epoch in range(epoch_begin, args.n_epochs):
        model.train()
        prev_lr = float(optimizer.param_groups[0]["lr"])
        lr_step = epoch // int(args.decay_epoch)
        batch_lr = float(args.beginning_lr) * (float(args.lr_decay_factor) ** lr_step)
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
                batch_cur, batch_msrc, batch_mdst, batch_ys, batch_yd, batch_yv = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_cur, batch_msrc, batch_mdst, batch_ys, batch_yd, batch_yv = next(train_iter)
            x_cur = batched_current_nhwc_to_torch(
                batch_cur, device, pin_memory=pin_mem, non_blocking=pin_mem
            )
            msrc = torch.as_tensor(batch_msrc, dtype=torch.bool)
            mdst = torch.as_tensor(batch_mdst, dtype=torch.bool)
            tgt_s = torch.as_tensor(batch_ys, dtype=torch.long)
            tgt_d = torch.as_tensor(batch_yd, dtype=torch.long)
            target_v = torch.as_tensor(batch_yv, dtype=torch.long)
            if pin_mem:
                msrc = msrc.pin_memory()
                mdst = mdst.pin_memory()
                tgt_s = tgt_s.pin_memory()
                tgt_d = tgt_d.pin_memory()
                target_v = target_v.pin_memory()
            msrc = msrc.to(device, non_blocking=pin_mem)
            mdst = mdst.to(device, non_blocking=pin_mem)
            tgt_s = tgt_s.to(device, non_blocking=pin_mem)
            tgt_d = tgt_d.to(device, non_blocking=pin_mem)
            target_v = target_v.to(device, non_blocking=pin_mem)

            src_oh = F.one_hot(tgt_s, num_classes=POLICY_GRID_NUMEL).float()
            optimizer.zero_grad(set_to_none=True)
            logits_s, logits_d, logits_v = model(x_cur, src_oh)
            loss_s = F.cross_entropy(logits_s.masked_fill(~msrc, -1e9), tgt_s)
            loss_d = F.cross_entropy(logits_d.masked_fill(~mdst, -1e9), tgt_d)
            loss_p = loss_s + loss_d
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
                acc_s = accuracy_from_logits_masked(logits_s, tgt_s, msrc)
                acc_d = accuracy_from_logits_masked(logits_d, tgt_d, mdst)
                acc_j = joint_move_accuracy(logits_s, logits_d, msrc, mdst, tgt_s, tgt_d)
                acc_v = value_accuracy_ignore(logits_v, target_v)

            global_step += 1
            writer.add_scalar("train/loss_policy_src", float(loss_s.item()), global_step)
            writer.add_scalar("train/loss_policy_dst", float(loss_d.item()), global_step)
            writer.add_scalar("train/loss_policy", float(loss_p.item()), global_step)
            writer.add_scalar("train/loss_total", float(loss.item()), global_step)
            writer.add_scalar("train/acc_src", float(acc_s.item()), global_step)
            writer.add_scalar("train/acc_dst", float(acc_d.item()), global_step)
            writer.add_scalar("train/acc_move_joint", float(acc_j.item()), global_step)
            writer.add_scalar("train/lr", batch_lr, global_step)
            # 无终局标签的 batch 里 acc_v 被定义为 0；不参与 TensorBoard / 进度条 EMA，否则会远低于随机三分类基线。
            if labeled_v.any():
                writer.add_scalar("train/loss_value", float(loss_v.item()), global_step)
                writer.add_scalar("train/acc_stm_outcome", float(acc_v.item()), global_step)
                expacc_v.update(float(acc_v.item()) * 100)
                exploss_v.update(float(loss_v.item()))

            expacc.update(float(acc_j.item()) * 100)
            exploss.update(float(loss.item()))
            out_pct = f"{expacc_v.getval()}%" if expacc_v.val is not None else "n/a"
            vloss_s = f"{exploss_v.getval()}" if exploss_v.val is not None else "n/a"
            pb.info = (
                f"EPOCH {epoch} STEP {batch_i} LR {batch_lr} "
                f"ACCjoint {expacc.getval()}% ACCout {out_pct} "
                f"LOSS {exploss.getval()} (v {vloss_s}) "
            )
            pb.complete(args.batch_size)
        print()

        model.eval()
        accs: list[float] = []
        losses: list[float] = []
        losses_p: list[float] = []
        losses_v: list[float] = []
        val_v_correct = 0
        val_v_total = 0
        pb = ProgressBar(worksum=args.n_batch_test * args.batch_size, info=f"validating epoch {epoch}")
        pb.startjob()
        with torch.no_grad():
            for _ in range(args.n_batch_test):
                try:
                    batch_cur, batch_msrc, batch_mdst, batch_ys, batch_yd, batch_yv = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    batch_cur, batch_msrc, batch_mdst, batch_ys, batch_yd, batch_yv = next(test_iter)
                x_cur = batched_current_nhwc_to_torch(
                    batch_cur, device, pin_memory=pin_mem, non_blocking=pin_mem
                )
                msrc = torch.as_tensor(batch_msrc, dtype=torch.bool).to(device, non_blocking=pin_mem)
                mdst = torch.as_tensor(batch_mdst, dtype=torch.bool).to(device, non_blocking=pin_mem)
                tgt_s = torch.as_tensor(batch_ys, dtype=torch.long).to(device, non_blocking=pin_mem)
                tgt_d = torch.as_tensor(batch_yd, dtype=torch.long).to(device, non_blocking=pin_mem)
                target_v = torch.as_tensor(batch_yv, dtype=torch.long).to(device, non_blocking=pin_mem)
                src_oh = F.one_hot(tgt_s, num_classes=POLICY_GRID_NUMEL).float()
                logits_s, logits_d, logits_v = model(x_cur, src_oh)
                loss_s = F.cross_entropy(logits_s.masked_fill(~msrc, -1e9), tgt_s)
                loss_d = F.cross_entropy(logits_d.masked_fill(~mdst, -1e9), tgt_d)
                loss_p = loss_s + loss_d
                labeled_v = target_v != VALUE_LABEL_IGNORE
                if labeled_v.any():
                    loss_v = F.cross_entropy(
                        logits_v, target_v, ignore_index=VALUE_LABEL_IGNORE
                    )
                else:
                    loss_v = (logits_v * 0).sum()
                loss = loss_p + args.value_loss_weight * loss_v
                acc_j = joint_move_accuracy(logits_s, logits_d, msrc, mdst, tgt_s, tgt_d)
                accs.append(float(acc_j.item()))
                losses.append(float(loss.item()))
                losses_p.append(float(loss_p.item()))
                if labeled_v.any():
                    losses_v.append(float(loss_v.item()))
                    m = labeled_v
                    pred = logits_v[m].argmax(dim=1)
                    val_v_correct += int((pred == target_v[m]).sum().item())
                    val_v_total += int(m.sum().item())
                pb.complete(args.batch_size)
        val_acc_v = (val_v_correct / val_v_total) if val_v_total else float("nan")
        val_loss_v = float(np.average(losses_v)) if losses_v else float("nan")
        acc_j_pct = 100.0 * float(np.average(accs))
        acc_v_pct = (100.0 * val_acc_v) if val_v_total else float("nan")
        print(
            f"TEST ACC(joint)={acc_j_pct:.2f}% "
            f"ACC(stm-out)={acc_v_pct:.2f}% "
            f"LOSS total={float(np.average(losses)):.4f} "
            f"pol={float(np.average(losses_p)):.4f} "
            f"val={val_loss_v:.4f}"
        )
        writer.add_scalar("val/loss_total", float(np.average(losses)), epoch)
        writer.add_scalar("val/loss_policy", float(np.average(losses_p)), epoch)
        if losses_v:
            writer.add_scalar("val/loss_value", val_loss_v, epoch)
        writer.add_scalar("val/acc_move_joint", float(np.average(accs)), epoch)
        if val_v_total:
            writer.add_scalar("val/acc_stm_outcome", float(val_acc_v), epoch)
        print()

        val_loss = float(np.average(losses))
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "num_res_layers": args.num_res_layers,
            "filters": model.filters,
            "in_channels": model.in_channels,
            "backbone": "resnet",
            "optim": "radam",
            "lr_decay_factor": float(args.lr_decay_factor),
            "decay_epoch": int(args.decay_epoch),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if improved:
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"  -> saved best.pt (val_loss={val_loss:.4f})")

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(
                f"[early-stop] 验证 total loss 已连续 {epochs_no_improve} 个 epoch 未优于 "
                f"best={best_val_loss:.4f}，停止训练（推理请优先用 best.pt）"
            )
            break

    writer.close()


if __name__ == "__main__":
    main()
