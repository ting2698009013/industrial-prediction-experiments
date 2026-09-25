#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_NAMES = ["f62", "f63", "f64"]


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


def compute_score(pred, true):
    if pred.shape != true.shape:
        raise ValueError(f"pred shape {pred.shape} != true shape {true.shape}")

    if pred.ndim != 3 or pred.shape[0] != 3:
        raise ValueError(
            f"answer 应该是 shape=(3,10,pred_len)，现在是 {pred.shape}"
        )

    n_targets, n_batches, pred_len = pred.shape

    batch_rows = []
    total_len = 0
    weighted_mse = {name: 0.0 for name in TARGET_NAMES}

    for bi in range(n_batches):
        row = {
            "batch": int(bi),
            "length": int(pred_len),
        }

        for ti, name in enumerate(TARGET_NAMES):
            err = pred[ti, bi].astype(np.float64) - true[ti, bi].astype(np.float64)

            mse = float(np.mean(err * err))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))

            row[f"mse_{name}"] = mse
            row[f"mae_{name}"] = mae
            row[f"bias_{name}"] = bias

            weighted_mse[name] += mse * pred_len

        row["mse_sum"] = float(
            row["mse_f62"] + row["mse_f63"] + row["mse_f64"]
        )

        total_len += pred_len
        batch_rows.append(row)

    summary = {}

    for name in TARGET_NAMES:
        summary[f"mse_{name}"] = float(weighted_mse[name] / max(1, total_len))

    summary["mse_sum"] = float(
        summary["mse_f62"] + summary["mse_f63"] + summary["mse_f64"]
    )

    flat_pred = pred.transpose(1, 2, 0).reshape(-1, 3)
    flat_true = true.transpose(1, 2, 0).reshape(-1, 3)

    for ti, name in enumerate(TARGET_NAMES):
        err = flat_pred[:, ti].astype(np.float64) - flat_true[:, ti].astype(np.float64)
        summary[f"mae_{name}"] = float(np.mean(np.abs(err)))
        summary[f"bias_{name}"] = float(np.mean(err))

    return summary, batch_rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--pred", required=True, help="生成的 answer.npy")
    p.add_argument("--true", required=True, help="pseudo_valid_answer.npy")
    p.add_argument("--out", default=None, help="score json")

    args = p.parse_args()

    pred = np.load(args.pred)
    true = np.load(args.true)

    summary, batch_rows = compute_score(pred, true)

    result = {
        "pred": args.pred,
        "true": args.true,
        "shape": list(pred.shape),
        "format": "answer[0]=f62, answer[1]=f63, answer[2]=f64",
        "summary": summary,
        "batch_metrics": batch_rows,
    }

    out = Path(args.out) if args.out else Path(args.pred).with_suffix(".score.json")
    save_json(result, out)

    print("[SCORE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[OK] saved:", out)


if __name__ == "__main__":
    main()