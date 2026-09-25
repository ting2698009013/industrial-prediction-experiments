#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkmm.config import SELECTED_COLS
from zkmm.features_direct import build_prediction_features_for_batch


def get_f64_bundle(bundle):
    if "target_2" in bundle:
        return bundle["target_2"]

    if "targets" in bundle and "target_2" in bundle["targets"]:
        return bundle["targets"]["target_2"]

    raise ValueError("f64 bundle 里找不到 target_2")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--answer", required=True, help="已有 answer.npy")
    p.add_argument("--question", required=True, help="question_2d.npy, shape=(N,65)")
    p.add_argument("--labels", required=True, help="mode labels, shape=(N,)")
    p.add_argument("--f64-bundle", required=True, help="f64_model_bundle.joblib")
    p.add_argument("--out", required=True)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)

    args = p.parse_args()

    answer = np.load(args.answer).astype(np.float32)
    question = np.load(args.question, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r").reshape(-1)

    if question.ndim != 2 or question.shape[1] != 65:
        raise ValueError(f"question should be (N,65), got {question.shape}")

    if len(labels) != len(question):
        raise ValueError(f"labels length {len(labels)} != question length {len(question)}")

    n = len(question)
    if n % args.n_batches != 0:
        raise ValueError(f"N={n} cannot divide n_batches={args.n_batches}")

    batch_len = n // args.n_batches
    visible_len = int(round(batch_len * args.visible_ratio))
    pred_len = batch_len - visible_len

    if answer.shape != (3, args.n_batches, pred_len):
        raise ValueError(
            f"answer shape {answer.shape} != expected {(3, args.n_batches, pred_len)}"
        )

    bundle = joblib.load(args.f64_bundle)
    tb = get_f64_bundle(bundle)
    model = tb["global_model"]

    pred_f64 = np.zeros((args.n_batches, pred_len), dtype=np.float32)

    for bi in range(args.n_batches):
        bs = bi * batch_len
        be = bs + batch_len

        batch = np.array(question[bs:be], dtype=np.float32, copy=True)
        labels_batch = np.asarray(labels[bs:be])

        X, names = build_prediction_features_for_batch(
            batch=batch,
            target_index=2,
            labels_batch=labels_batch,
            selected_cols=SELECTED_COLS,
            add_mode=True,
            add_switch=tb.get("add_switch", True),
            cross=tb.get("cross", "core"),
            f63_state="base",
        )

        p64 = model.predict(X).astype(np.float32)

        if p64.shape[0] != pred_len:
            raise ValueError(f"batch {bi}: p64 len {len(p64)} != pred_len {pred_len}")

        pred_f64[bi] = p64

        print(
            f"[F64-ML] batch={bi}, pred min={float(p64.min()):.6f}, "
            f"max={float(p64.max()):.6f}, mean={float(p64.mean()):.6f}",
            flush=True,
        )

    answer[2] = pred_f64

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, answer.astype(np.float32))

    print("[OK] saved:", out)
    print("[OK] shape:", answer.shape)
    print("[CHECK] f64 range:", float(answer[2].min()), float(answer[2].max()))


if __name__ == "__main__":
    main()