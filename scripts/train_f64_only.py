#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkmm.data import load_train, save_json
from zkmm.train_direct import train_target_model


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)

    p.add_argument("--model-dir", default="runs/f64_only_lgbm")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat", "histgb"])

    p.add_argument("--fast", action="store_true")
    p.add_argument("--cut-stride", type=int, default=3000)
    p.add_argument("--row-stride", type=int, default=5)

    # f64 建议 core 或 full；full 特征更多，可能更强但慢一点
    p.add_argument("--cross", default="core", choices=["none", "core", "full"])

    args = p.parse_args()

    data = load_train(args.data, mmap=True)
    labels = np.load(args.labels)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[INFO] training f64 only")
    print("[INFO] target_index=2 means f64")
    print("[INFO] data:", args.data)
    print("[INFO] labels:", args.labels)
    print("[INFO] model_dir:", out)
    print("[INFO] cross:", args.cross)

    model, meta = train_target_model(
        data=data,
        target_index=2,
        train_end=len(data),
        labels=labels,
        mode_filter=None,
        model_name=args.model,
        fast=args.fast,
        cut_stride=args.cut_stride,
        row_stride=args.row_stride,
        cross=args.cross,
        add_switch=True,
        f63_state="base",
    )

    target_2 = {
        "global_model": model,
        **meta,
        "blend_global_weight": 0.0,
        "kind": "f64_direct_model",
        "target_name": "f64",
        "target_index": 2,
        "formula": "pred_f64 = model_predict(features)",
    }

    bundle = {
        "meta": {
            "kind": "f64_only_bundle",
            "data": args.data,
            "labels": args.labels,
            "model": args.model,
            "fast": bool(args.fast),
            "cut_stride": int(args.cut_stride),
            "row_stride": int(args.row_stride),
            "cross": args.cross,
        },
        "targets": {
            "target_2": target_2,
        },
        "target_2": target_2,
    }

    joblib.dump(bundle, out / "f64_model_bundle.joblib")

    summary = {
        "model_dir": str(out),
        "model_file": "f64_model_bundle.joblib",
        "target": "f64",
        "target_index": 2,
        "n_samples": meta.get("n_samples"),
        "n_features": meta.get("n_features"),
        "fit_seconds": meta.get("fit_seconds"),
        "model": args.model,
        "fast": bool(args.fast),
        "cut_stride": int(args.cut_stride),
        "row_stride": int(args.row_stride),
        "cross": args.cross,
        "feature_names": meta.get("feature_names"),
    }

    save_json(summary, out / "summary.json")

    print("[OK] saved:", out / "f64_model_bundle.joblib")
    print("[OK] saved:", out / "summary.json")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()