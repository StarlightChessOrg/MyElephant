# MyElephant

在 [Icy Elephant](https://github.com/bupticybee/icyElephant) 思路上整理的中国象棋 **PyTorch 策略 + 价值训练 / 图形对弈** 代码：

- **策略头（与 UI 一致：先选子再选落点）**：仅用走棋前当前局面的 **7 路有符号兵种平面**（红 +1 / 黑 −1，固定红方棋盘视角）过共享卷积主干 → **起点格** 90 类（ICCS 展平 `y*9+x`）+ 在 teacher/推理给定起点 one-hot 后 **落点格** 90 类；训练时对起点、落点各做交叉熵（落点 mask 为「在棋谱真实起点下」的合法到达格）。
- **价值头**：同一主干池化后输出红方 **胜 / 和 / 负** 三分类 logit，与棋谱 `Head/RecordResult` 对齐做交叉熵（无标签样本忽略）。
- 走棋方不进平面编码；MCTS / 纯网络走子用分解式 `P(着)=P(起点)P(落点|起点)` 在合法着集合上归一化得到 prior。

## 目录

| 路径 | 说明 |
|------|------|
| `cchess/` | 规则引擎与棋谱读取（GPL，与上游一致）。 |
| `my_elephant/chess/` | 盘面编码、`convert_game`（两阶段标签）、`iccs_flat_index`、`src_dst_masks_and_labels`、`successor_planes_for_legals`（遗留）、`GamePlay` 等。 |
| `my_elephant/datasets/` | `ProgressBar` 等训练辅助。 |
| `my_elephant/data_prep/` | 生成 train/test 路径清单等 CLI（`split_manifest`）。 |
| `my_elephant/training/` | `SuccessorPolicy`（起点+落点+价值）、`policy_data`、`mcts_engine`（PUCT MCTS）、`train_policy_torch` / `play_policy_torch`。 |

推荐：`from my_elephant import GamePlay, convert_game, successor_planes_for_legals`。

## 安装

在仓库根目录（与 `pyproject.toml` 同级）：

```bash
pip install -e ".[training]"
```

## 数据

本仓库中的 **`.cbf`** 为 **XML**（根元素 `ChineseChessRecord`），含 `Head/FEN`、`MoveList`、`RecordResult` 等。`RecordResult` 与 `cchess.reader_xqf` 中约定一致：`1` 红胜、`2` 黑胜、`3`/`4` 和、`0` 或缺失表示无可靠终局标签（价值损失自动忽略）。

### 方式一：清单 CSV

默认读取 `data/train_list.csv`、`data/test_list.csv`（单列、无表头，每行一个棋谱路径）。

可用脚本从目录生成清单：

```bash
python -m my_elephant.data_prep.split_manifest --cbf-dir 你的cbf目录
```

### 方式二：训练时直接指定根目录（推荐）

训练时加 **`--cbf-root`**，在目录下（默认递归）搜集 `.cbf`，按 `--train-ratio` / `--data-seed` 划分 train/val，**不再读取** `--train-list` / `--test-list`。仅搜一层目录时加 **`--cbf-shallow`**。

## 训练

```bash
cd /path/to/MyElephant
python -m my_elephant.training.train_policy_torch --model-name my_run
```

**续训**：与上次相同的 `--model-name`（及 `--model-dir` 若改过）时，加 **`--continue`** 即可自动从 **`models/<model-name>/last.pt`** 恢复权重、优化器、epoch、`global_step`、`best_val_loss`；若该文件不存在会提示并从头训练。仍可用 **`--resume path/to.pt`** 指定任意 checkpoint；**同时写 `--resume` 与 `--continue` 时只认 `--resume`**。

```bash
python -m my_elephant.training.train_policy_torch --model-name my_run --continue
```

### 默认模型规模（便于 MCTS 多次前向）

- **`--num-res-layers`** 默认 **4**，**`--filters`** 默认 **64**（权重约数 MB 量级；旧 checkpoint 请与训练时一致或从 ckpt 自动推断）。
- Checkpoint 中保存 **`filters`**、**`num_res_layers`**；`--resume` 时会与权重形状对齐。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--cbf-root` | 数据集根目录，自动搜 `.cbf` 并划分 train/val |
| `--batch-size` | 每步 batch 大小 |
| `--num-workers` | `DataLoader` 子进程数；**默认** `min(8, CPU 核数)`，`0` 表示主进程加载 |
| `--prefetch-factor` | 每 worker 预取 batch 数（`num_workers>0` 时） |
| `--value-loss-weight` | 价值头 CE 相对策略损失（起点 CE + 落点 CE）的权重（默认 `0.5`） |
| `--continue` | 续训：自动加载 `models/<model-name>/last.pt`（见上文） |
| `--resume` | 从指定 `.pt` 恢复；权重在 **CPU** 上加载后再 `model.to(device)`；与 `--continue` 同时存在时优先本项 |

数据管线使用 **`torch.utils.data.DataLoader` + `IterableDataset`**：多 worker 按文件列表分片并行读盘与解析；`worker_init_fn` 限制各进程 BLAS 线程数；CUDA 时可传 **`pin_memory_device`**；训练集默认 **`drop_last=True`**。

也可使用入口：`my-train-policy --model-name my_run`。

## 对弈（Tkinter）

`play_policy_torch` 为 **图形界面**：圆形棋子、鼠标选子再走子；**箭头**标示上一手起点→终点；红/黑可分别选择 **人类**、**纯网络**、**MCTS+策略价值网络**。MCTS 在后台线程运行，避免卡 UI。

```bash
python -m my_elephant.training.play_policy_torch --checkpoint models/my_run/best.pt
```

常用参数：`--gpu`、**`--mcts-sims`**（MCTS 模拟次数，默认 `320`）、**`--c-puct`**（PUCT 系数，默认 `1.5`）。

或：`my-play-policy --checkpoint ...`。

## 说明

- 原始数据链接可参考 Icy Elephant README / 百度盘说明；本仓库不附带棋谱二进制。
- 若 `cchess` 在 Python 3 下导入失败（如 `sets` 模块），需按 Python 3 兼容性修补 `cchess/piece.py` 等文件。
- 网上常把象棋桥 **`.cbr`/`.cbl` 二进制** 称作棋谱格式；与本仓库 **XML `.cbf`** 不是同一格式。
