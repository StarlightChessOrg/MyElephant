# MyElephant

在 [Icy Elephant](https://github.com/bupticybee/icyElephant) 思路上整理的中国象棋 **PyTorch 策略训练/对弈** 代码：合法着法枚举 → 每个着法的**走后局面** → **7 路有符号兵种输入**（红 +1 / 黑 −1，固定红方棋盘视角）→ 单头 logit；训练用棋谱着法做交叉熵；走棋方不进网络，用 `logits_as_red_preference`（红 `argmax` raw、黑 `argmax(-raw)`）与损失对齐。

## 目录

| 路径 | 说明 |
|------|------|
| `cchess/` | 规则引擎与棋谱读取（GPL，与上游一致）。 |
| `my_elephant/chess/` | 盘面编码、`convert_game`、`successor_planes_for_legals`、`GamePlay` 等。 |
| `my_elephant/datasets/` | `ProgressBar` 等训练辅助。 |
| `my_elephant/data_prep/` | 生成 train/test 路径清单等 CLI。 |
| `my_elephant/training/` | `SuccessorPolicy`、DataLoader、train / play 入口。 |

推荐：`from my_elephant import GamePlay, convert_game, successor_planes_for_legals`。

## 安装

在仓库根目录（与 `pyproject.toml` 同级）：

```bash
pip install -e ".[training]"
```

## 数据清单

默认读取 `data/train_list.csv`、`data/test_list.csv`（单列、无表头，每行一个棋谱 XML 路径）。可用：

```bash
python -m my_elephant.data_prep.split_manifest --cbf-dir 你的cbf目录
```

## 训练

```bash
cd /path/to/MyElephant
python -m my_elephant.training.train_policy_torch --model-name my_run --num-res-layers 10
```

常用参数：`--batch-size`、`--num-workers`、`--prefetch-factor`、`--resume models/my_run/last.pt`。

也可使用入口：`my-train-policy --model-name my_run`。

## 对弈

```bash
python -m my_elephant.training.play_policy_torch --checkpoint models/my_run/best.pt
```

或：`my-play-policy --checkpoint ...`。

## 说明

- 原始数据链接可参考 Icy Elephant README / 百度盘说明；本仓库不附带棋谱二进制。
- 若 `cchess` 在 Python 3 下导入失败（如 `sets` 模块），需按 Python 3 兼容性修补 `cchess/piece.py` 等文件。
