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

    p.add_argument("--data", required=True, help="clean_train_wcj.npy")
    p.add_argument("--labels", required=True, help="mode_labels.npy")

    p.add_argument("--model-dir", default="runs/f62_only_model")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat", "histgb"])

    p.add_argument("--fast", action="store_true")
    p.add_argument("--cut-stride", type=int, default=3000)
    p.add_argument("--row-stride", type=int, default=5)

    # 对 f62 来说 cross / f63-state 实际不会产生作用，
    # 但保留参数，方便和原 train_target_model 接口一致。
    p.add_argument("--cross", default="core", choices=["none", "core", "full"])
    p.add_argument("--f63-state", default="base", choices=["base", "strong"])

    args = p.parse_args()

    data = load_train(args.data, mmap=True)
    labels = np.load(args.labels)

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[INFO] training f62 only")
    print("[INFO] target_index=0 means residual target: f62 - f01")
    print("[INFO] data:", args.data)
    print("[INFO] labels:", args.labels)
    print("[INFO] model_dir:", out)

    # 只训练 target_index=0，也就是 f62 residual
    model, meta = train_target_model(
        data=data,
        target_index=0,
        train_end=len(data),
        labels=labels,
        mode_filter=None,
        model_name=args.model,
        fast=args.fast,
        cut_stride=args.cut_stride,
        row_stride=args.row_stride,
        cross=args.cross,
        add_switch=True,
        f63_state=args.f63_state,
    )

    target_0 = {
        "global_model": model,
        **meta,
        "blend_global_weight": 0.0,
        "kind": "f62_residual_model",
        "target_name": "f62",
        "target_index": 0,
        "formula": "pred_f62 = f01_future + model_predict(f62_minus_f01_residual)",
    }

    # 保存成单目标 bundle，后面生成 submit 时只取 target_0
    bundle = {
        "meta": {
            "kind": "f62_only_residual_bundle",
            "data": args.data,
            "labels": args.labels,
            "model": args.model,
            "fast": bool(args.fast),
            "cut_stride": int(args.cut_stride),
            "row_stride": int(args.row_stride),
            "cross": args.cross,
            "f63_state": args.f63_state,
        },
        "targets": {
            "target_0": target_0,
        },

        # 兼容后续脚本直接 bundle["target_0"] 读取
        "target_0": target_0,
    }

    joblib.dump(bundle, out / "f62_model_bundle.joblib")

    summary = {
        "model_dir": str(out),
        "model_file": "f62_model_bundle.joblib",
        "target": "f62",
        "target_index": 0,
        "train_target": "f62 - f01",
        "restore_formula": "pred_f62 = f01_future + predicted_residual",
        "n_samples": meta.get("n_samples"),
        "n_features": meta.get("n_features"),
        "fit_seconds": meta.get("fit_seconds"),
        "model": args.model,
        "fast": bool(args.fast),
        "cut_stride": int(args.cut_stride),
        "row_stride": int(args.row_stride),
        "cross": args.cross,
        "f62_res_q001": meta.get("f62_res_q001"),
        "f62_res_q999": meta.get("f62_res_q999"),
        "feature_names": meta.get("feature_names"),
    }

    save_json(summary, out / "summary.json")

    print("[OK] saved:", out / "f62_model_bundle.joblib")
    print("[OK] saved:", out / "summary.json")
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()