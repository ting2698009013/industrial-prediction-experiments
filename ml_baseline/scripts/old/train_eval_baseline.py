from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkml.config import TARGET_NAMES, HISTORY_LENGTH, BATCH_LENGTH, PRED_LENGTH
from zkml.data import load_clean_data, load_labels, fold_ids, fold_slices
from zkml.features import build_training_matrix
from zkml.infer import recursive_predict_batch
from zkml.modeling import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cleaned full data, shape=(600000,65)")
    ap.add_argument("--labels", default=None, help="optional mode_labels.npy")
    ap.add_argument("--folds", default="9", help="0..9, comma list, or all")
    ap.add_argument("--model", default="auto", choices=["auto", "lgbm", "xgb", "hgb", "ridge"])
    ap.add_argument("--fast", action="store_true", help="use fewer estimators")
    ap.add_argument("--sample-stride", type=int, default=1, help="use >1 for faster training")
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_clean_data(args.data)
    labels = load_labels(args.labels, len(full))
    ids = fold_ids(args.folds)
    rows = []

    print(f"[LOAD] full={full.shape}, labels={None if labels is None else labels.shape} (None 表示不使用模式one-hot特征)", flush=True)
    print(f"[RUN] folds={ids}, model={args.model}, stride={args.sample_stride}", flush=True)

    for bid in ids:
        b0, val_start, val_end, _ = fold_slices(bid)
        print(f"\n========== fold/batch {bid}: val=[{val_start},{val_end}) ==========", flush=True)
        fold_t0 = time.time()
        models = []
        for ti, name in enumerate(TARGET_NAMES):
            t0 = time.time()
            Xtr, ytr, names = build_training_matrix(
                full, ti, train_end=val_start, labels=labels, sample_stride=args.sample_stride
            )
            model = make_model(args.model, seed=42 + ti, fast=args.fast)
            print(f"  train {name}: X={Xtr.shape}, y={ytr.shape}", flush=True)
            model.fit(Xtr, ytr)
            models.append(model)
            print(f"  done {name}: {time.time() - t0:.1f}s", flush=True)

        batch = full[b0:b0 + BATCH_LENGTH].copy()
        # 模拟平台：后 20% 目标列置 0，但非目标输入保持可见
        answer = batch[HISTORY_LENGTH:, 62:65].T.copy()
        batch[HISTORY_LENGTH:, 62:65] = 0.0
        lab_batch = labels[b0:b0 + BATCH_LENGTH] if labels is not None else None
        pred = recursive_predict_batch(batch, models, labels=lab_batch)
        per_target = ((pred - answer) ** 2).mean(axis=1)
        total = float(per_target.sum())
        print(f"  MSE per target: f62={per_target[0]:.8f}, f63={per_target[1]:.8f}, f64={per_target[2]:.8f}, sum={total:.8f}", flush=True)
        np.save(out_dir / f"pred_fold{bid}.npy", pred.astype(np.float32))
        rows.append({
            "fold": bid,
            "mse_f62": float(per_target[0]),
            "mse_f63": float(per_target[1]),
            "mse_f64": float(per_target[2]),
            "mse_sum": total,
            "seconds": round(time.time() - fold_t0, 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cv_summary.csv", index=False)
    summary = {
        "folds": ids,
        "mean_mse_f62": float(df["mse_f62"].mean()),
        "mean_mse_f63": float(df["mse_f63"].mean()),
        "mean_mse_f64": float(df["mse_f64"].mean()),
        "mean_mse_sum": float(df["mse_sum"].mean()),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== SUMMARY ==========", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
