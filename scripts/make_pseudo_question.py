#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import json


TARGET_COLS = [62, 63, 64]


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_question_and_answer(arr, n_batches=10, visible_ratio=0.8):
    """
    输入:
      arr: shape = (sequence_length, 65)

    输出:
      question:
        shape = (sequence_length, 65)
        每个 batch 后 20% 的 f62/f63/f64 置 0

      answer:
        shape = (3, n_batches, pred_len)
        保存每个 batch 后 20% 的真实 f62/f63/f64
    """
    if arr.ndim != 2 or arr.shape[1] != 65:
        raise ValueError(f"expected arr shape (N,65), got {arr.shape}")

    n = arr.shape[0]
    if n % n_batches != 0:
        raise ValueError(f"sequence_length={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    if pred_len <= 0:
        raise ValueError(f"pred_len={pred_len} 非法，请检查 visible_ratio={visible_ratio}")

    question = arr.copy().astype(np.float32)
    answer = np.zeros((3, n_batches, pred_len), dtype=np.float32)

    windows = []

    for bi in range(n_batches):
        bs = bi * batch_len
        cut = bs + visible_len
        be = bs + batch_len

        answer[:, bi, :] = arr[cut:be, TARGET_COLS].T.astype(np.float32)

        question[cut:be, TARGET_COLS] = 0.0

        windows.append({
            "batch": bi,
            "batch_start": int(bs),
            "visible_start": int(bs),
            "visible_end": int(cut),
            "pred_start": int(cut),
            "pred_end": int(be),
            "batch_end": int(be),
            "batch_len": int(batch_len),
            "visible_len": int(visible_len),
            "pred_len": int(pred_len),
        })

    return question, answer, windows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True, help="clean_train_wcj.npy")
    p.add_argument("--out", required=True, help="output directory")

    # 默认 80% 做 pseudo train，20% 做 pseudo validation
    # 对 600000 来说：train=480000, valid=120000
    p.add_argument("--train-ratio", type=float, default=0.8)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)

    # 可选：如果你已经有全量 mode_labels.npy，也顺手切成两份
    p.add_argument("--labels", default=None, help="optional full mode_labels.npy")

    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.data)
    if arr.ndim != 2 or arr.shape[1] != 65:
        raise ValueError(f"expected data shape (N,65), got {arr.shape}")

    n = arr.shape[0]
    split = int(round(n * args.train_ratio))

    # 为了保证 pseudo train 和 pseudo valid 都能平均切成 10 个 batch，
    # split 需要同时满足 train_len 和 valid_len 都能被 n_batches 整除。
    train_len = split
    valid_len = n - split

    if train_len % args.n_batches != 0 or valid_len % args.n_batches != 0:
        raise ValueError(
            f"当前 split 不合适: train_len={train_len}, valid_len={valid_len}, "
            f"都需要能被 n_batches={args.n_batches} 整除。"
            f"建议 train-ratio=0.8，对 600000 会得到 480000/120000。"
        )

    train_arr = arr[:train_len].astype(np.float32)
    valid_full = arr[train_len:].astype(np.float32)

    valid_question, valid_answer, valid_windows = make_question_and_answer(
        valid_full,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
    )

    np.save(out / "pseudo_train.npy", train_arr)
    np.save(out / "pseudo_valid_full.npy", valid_full)
    np.save(out / "pseudo_valid_question.npy", valid_question)
    np.save(out / "pseudo_valid_answer.npy", valid_answer)

    labels_info = None
    if args.labels:
        labels = np.load(args.labels)
        if labels.ndim == 2:
            labels = labels.reshape(-1)

        if len(labels) != n:
            raise ValueError(f"labels length {len(labels)} != data length {n}")

        train_labels = labels[:train_len].astype(np.int16)
        valid_labels = labels[train_len:].astype(np.int16)

        np.save(out / "pseudo_train_mode_labels.npy", train_labels)
        np.save(out / "pseudo_valid_mode_labels.npy", valid_labels)

        labels_info = {
            "labels": args.labels,
            "pseudo_train_mode_labels": "pseudo_train_mode_labels.npy",
            "pseudo_valid_mode_labels": "pseudo_valid_mode_labels.npy",
        }

    meta = {
        "source": args.data,
        "shape": list(arr.shape),
        "train_ratio": args.train_ratio,
        "n_batches": args.n_batches,
        "visible_ratio": args.visible_ratio,
        "train": {
            "file": "pseudo_train.npy",
            "shape": list(train_arr.shape),
            "sequence_range": [0, int(train_len)],
        },
        "valid_full": {
            "file": "pseudo_valid_full.npy",
            "shape": list(valid_full.shape),
            "sequence_range": [int(train_len), int(n)],
        },
        "valid_question": {
            "file": "pseudo_valid_question.npy",
            "shape": list(valid_question.shape),
            "note": "每个 batch 后 20% 的 f62/f63/f64 已置 0",
        },
        "valid_answer": {
            "file": "pseudo_valid_answer.npy",
            "shape": list(valid_answer.shape),
            "target_order": TARGET_COLS,
        },
        "valid_windows": valid_windows,
        "labels_info": labels_info,
    }

    save_json(meta, out / "pseudo_split_meta.json")

    print("[OK] pseudo split saved to:", out)
    print("[TRAIN]", train_arr.shape, "->", out / "pseudo_train.npy")
    print("[VALID QUESTION]", valid_question.shape, "->", out / "pseudo_valid_question.npy")
    print("[VALID ANSWER]", valid_answer.shape, "->", out / "pseudo_valid_answer.npy")
    if args.labels:
        print("[LABELS] train/valid mode labels saved")


if __name__ == "__main__":
    main()