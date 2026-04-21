"""从 data/imsa 下的 JSON 列表中提取 playbook_id（原 get_data.ipynb 逻辑）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_playbook_ids(data_prefix: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(data_prefix.iterdir()):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as f:
                onejson = json.load(f)
        except (OSError, json.JSONDecodeError):
            print("跳过无法解析的文件:", path)
            continue
        try:
            playlist = onejson["response"]["list"]
        except (KeyError, TypeError):
            continue
        for play in playlist:
            try:
                ids.append(int(play["playbook_id"]))
            except (KeyError, TypeError, ValueError):
                print("一条记录缺少 playbook_id，已跳过")
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="收集 IMSA playbook_id 列表")
    p.add_argument("--data-prefix", type=Path, default=Path("data/imsa"))
    p.add_argument("--out", type=Path, default=Path("data/playbook_ids.txt"), help="每行一个 id")
    args = p.parse_args()
    ids = collect_playbook_ids(args.data_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(i) for i in ids), encoding="utf-8")
    print(f"共 {len(ids)} 个 id（唯一值 {len(set(ids))}），已写入 {args.out}")


if __name__ == "__main__":
    main()
