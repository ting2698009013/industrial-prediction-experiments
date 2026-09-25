#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"

for p in [SCRIPTS, SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_chunk1200_joint import train_chunk_model, parse_weights


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--question", required=True, help="data\\official\\question_2d.npy")
    p.add_argument("--labels", required=True, help="runs\\official_modes\\mode_labels.npy")
    p.add_argument("--out", required=True)

    p.add_argument("--init-ckpt", default=None)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--history-len", type=int, default=16384)
    p.add_argument("--chunk-len", type=int, default=1200)
    p.add_argument("--samples-per-batch", type=int, default=64)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--loss-weights", default="1.0,1.0")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--near-switch-win", type=int, default=512)

    # 默认：
    #   from pretrain 使用 ckpt scaler
    #   visible only 重新 fit visible scaler
    p.add_argument("--no-init-scalers", action="store_true")

    args = p.parse_args()

    if args.lr is None:
        if args.init_ckpt:
            lr = 5e-5
        else:
            lr = 1e-3
    else:
        lr = args.lr

    print("[MODE]", "from_pretrain" if args.init_ckpt else "visible_only")
    print("[LR]", lr)

    train_chunk_model(
        data_path=args.question,
        labels_path=args.labels,
        out_dir=args.out,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        history_len=args.history_len,
        chunk_len=args.chunk_len,
        samples_per_batch=args.samples_per_batch,

        # 核心：fine-tune 只使用每个 batch 前 80% 可见段
        train_visible_only=True,
        scaler_visible_only=True,

        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=lr,
        hidden=args.hidden,
        levels=args.levels,
        dropout=args.dropout,
        loss_weights=parse_weights(args.loss_weights),
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        near_switch_win=args.near_switch_win,
        init_ckpt=args.init_ckpt,
        use_init_scalers=not args.no_init_scalers,
    )


if __name__ == "__main__":
    main()