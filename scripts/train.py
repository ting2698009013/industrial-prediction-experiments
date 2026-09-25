#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from zkdl.train import train_official_windows
from zkdl.utils import save_json


def parse_weights(s):
    parts = str(s).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("--loss-weights 需要两个数，例如 1.0,1.0 或 1.0 1.0")
    return float(parts[0]), float(parts[1])


def main():
    p = argparse.ArgumentParser()

    # 目前先只保留 final，避免 fold/folds 入口继续引发函数名不匹配
    p.add_argument("--run", default="final", choices=["final"])

    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)

    p.add_argument("--history-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--loss-weights", default="1.0,1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--near-switch-win", type=int, default=512)
    p.add_argument("--init-ckpt", default=None)

    args = p.parse_args()

    data = np.load(args.data, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loss_weights = parse_weights(args.loss_weights)

    best = train_official_windows(
        data=data,
        labels=labels,
        out_dir=out,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        history_len=args.history_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        levels=args.levels,
        dropout=args.dropout,
        loss_weights=loss_weights,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        near_switch_win=args.near_switch_win,
        init_ckpt=args.init_ckpt,
    )

    save_json(
        {
            "run": "final",
            "data": args.data,
            "labels": args.labels,
            "out": args.out,
            "best": best,
            "loss_weights": list(loss_weights),
            "n_batches": args.n_batches,
            "visible_ratio": args.visible_ratio,
            "history_len": args.history_len,
            "hidden": args.hidden,
            "levels": args.levels,
            "dropout": args.dropout,
            "init_ckpt": args.init_ckpt,
        },
        out / "run_summary.json",
    )


if __name__ == "__main__":
    main()