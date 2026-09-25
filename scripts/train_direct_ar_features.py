#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkdl.train_direct_ar_features import train_official_windows


def parse_loss_weights(s):
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--loss-weights 格式应为 1.0,1.0")
    return float(parts[0]), float(parts[1])


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()

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

    loss_weights = parse_loss_weights(args.loss_weights)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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

    run_summary = {
        "run": args.run,
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
        "model_kind": "direct_ar_features",
    }

    save_json(run_summary, out / "run_summary.json")

    print("[DONE] best:", best)


if __name__ == "__main__":
    main()