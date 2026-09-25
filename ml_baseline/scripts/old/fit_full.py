from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkml.config import TARGET_NAMES
from zkml.data import load_clean_data, load_labels
from zkml.features import build_training_matrix
from zkml.modeling import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model", default="auto", choices=["auto", "lgbm", "xgb", "hgb", "ridge"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--sample-stride", type=int, default=1)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    full = load_clean_data(args.data)
    labels = load_labels(args.labels, len(full))
    meta = {"model": args.model, "sample_stride": args.sample_stride, "targets": TARGET_NAMES}

    for ti, name in enumerate(TARGET_NAMES):
        t0 = time.time()
        Xtr, ytr, fnames = build_training_matrix(full, ti, train_end=len(full), labels=labels, sample_stride=args.sample_stride)
        model = make_model(args.model, seed=42 + ti, fast=args.fast)
        print(f"[TRAIN] {name}: X={Xtr.shape}, y={ytr.shape}", flush=True)
        model.fit(Xtr, ytr)
        dump(model, model_dir / f"model_{name}.joblib")
        print(f"[SAVE] {name}: {time.time() - t0:.1f}s", flush=True)
        meta[f"features_{name}"] = len(fnames)

    # 保存训练集模式标签，用于线下复现。真实测试若没有 labels，可先不用 mode 特征，或另写 GMM detector。
    if labels is not None:
        import numpy as np
        np.save(model_dir / "mode_labels_train.npy", labels)
        meta["uses_mode_labels"] = True
    else:
        meta["uses_mode_labels"] = False
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
