#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_COLS = [62, 63, 64]


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


def make_question_and_answer(arr, n_batches=10, visible_ratio=0.8):
    """
    arr:
      shape = (N, 65)

    question:
      shape = (N, 65)
      每个 batch 后 20% 的 f62/f63/f64 置 0

    answer:
      shape = (3, n_batches, pred_len)
      answer[0] = f62
      answer[1] = f63
      answer[2] = f64
    """
    if arr.ndim != 2 or arr.shape[1] != 65:
        raise ValueError(f"expected arr shape (N,65), got {arr.shape}")

    n = arr.shape[0]
    if n % n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    if visible_len <= 0 or pred_len <= 0:
        raise ValueError(
            f"visible_len={visible_len}, pred_len={pred_len} 非法，"
            f"请检查 visible_ratio={visible_ratio}"
        )

    question = arr.astype(np.float32).copy()
    answer = np.zeros((3, n_batches, pred_len), dtype=np.float32)

    windows = []

    for bi in range(n_batches):
        bs = bi * batch_len
        cut = bs + visible_len
        be = bs + batch_len

        answer[:, bi, :] = arr[cut:be, TARGET_COLS].T.astype(np.float32)
        question[cut:be, TARGET_COLS] = 0.0

        windows.append(
            {
                "batch": int(bi),
                "batch_start": int(bs),
                "visible_start": int(bs),
                "visible_end": int(cut),
                "pred_start": int(cut),
                "pred_end": int(be),
                "batch_end": int(be),
                "batch_len": int(batch_len),
                "visible_len": int(visible_len),
                "pred_len": int(pred_len),
            }
        )

    return question, answer, windows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)

    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)

    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.data)
    if arr.ndim != 2 or arr.shape[1] != 65:
        raise ValueError(f"expected data shape (N,65), got {arr.shape}")

    n = arr.shape[0]
    split = int(round(n * args.train_ratio))

    train_len = split
    valid_len = n - split

    if train_len % args.n_batches != 0:
        raise ValueError(
            f"train_len={train_len} 不能被 n_batches={args.n_batches} 整除"
        )

    if valid_len % args.n_batches != 0:
        raise ValueError(
            f"valid_len={valid_len} 不能被 n_batches={args.n_batches} 整除"
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

    # 标准答案格式，和最终提交格式一致
    np.save(out / "pseudo_valid_answer.npy", valid_answer)

    labels_info = None

    if args.labels:
        labels = np.load(args.labels)
        labels = labels.reshape(-1)

        if len(labels) != n:
            raise ValueError(f"labels length {len(labels)} != data length {n}")

        train_labels = labels[:train_len].astype(np.int16)
        valid_labels = labels[train_len:].astype(np.int16)

        np.save(out / "pseudo_train_mode_labels.npy", train_labels)
        np.save(out / "pseudo_valid_mode_labels.npy", valid_labels)

        labels_info = {
            "source_labels": args.labels,
            "pseudo_train_mode_labels": "pseudo_train_mode_labels.npy",
            "pseudo_valid_mode_labels": "pseudo_valid_mode_labels.npy",
        }

    meta = {
        "source": args.data,
        "shape": list(arr.shape),
        "train_ratio": float(args.train_ratio),
        "n_batches": int(args.n_batches),
        "visible_ratio": float(args.visible_ratio),
        "target_order": ["f62", "f63", "f64"],
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
            "format": "answer[0]=f62, answer[1]=f63, answer[2]=f64",
        },
        "valid_windows": valid_windows,
        "labels_info": labels_info,
    }

    save_json(meta, out / "pseudo_split_meta.json")

    print("[OK] saved pseudo split to:", out)
    print("[TRAIN]", train_arr.shape)
    print("[VALID QUESTION]", valid_question.shape)
    print("[VALID ANSWER]", valid_answer.shape)
    print("[ANSWER FORMAT] answer[0]=f62, answer[1]=f63, answer[2]=f64")


if __name__ == "__main__":
    main()