from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zkml.config import BATCHES, BATCH_LENGTH, HISTORY_LENGTH, PRED_LENGTH, TARGET_NAMES
from zkml.data import load_clean_data


def main():
    ap = argparse.ArgumentParser(description="用完整训练集答案窗口评估 submit.npy，并定位误差集中在哪些 batch/目标/时间段。")
    ap.add_argument("--data", required=True, help="full data, shape=(600000,65)")
    ap.add_argument("--submit", required=True, help="submit/pred npy, shape=(3,10,12000)")
    ap.add_argument("--out", default="runs/submit_score")
    ap.add_argument("--window", type=int, default=1000, help="误差分段窗口长度，默认1000")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = load_clean_data(args.data)
    pred = np.load(args.submit).astype(np.float32, copy=False)
    if pred.shape != (3, BATCHES, PRED_LENGTH):
        raise ValueError(f"submit shape should be {(3,BATCHES,PRED_LENGTH)}, got {pred.shape}")

    # answer: (3,10,12000)
    ans = full.reshape(BATCHES, BATCH_LENGTH, 65)[:, HISTORY_LENGTH:, 62:65].transpose(2,0,1).astype(np.float32)
    err = pred - ans
    mse_tb = (err ** 2).mean(axis=2)  # (3,10)
    mae_tb = np.abs(err).mean(axis=2)
    bias_tb = err.mean(axis=2)

    rows = []
    for ti, name in enumerate(TARGET_NAMES):
        for b in range(BATCHES):
            rows.append({
                "target": name,
                "batch": b,
                "mse": float(mse_tb[ti, b]),
                "rmse": float(np.sqrt(mse_tb[ti, b])),
                "mae": float(mae_tb[ti, b]),
                "bias_pred_minus_true": float(bias_tb[ti, b]),
                "true_mean": float(ans[ti,b].mean()),
                "pred_mean": float(pred[ti,b].mean()),
                "true_std": float(ans[ti,b].std()),
                "pred_std": float(pred[ti,b].std()),
                "true_start": float(ans[ti,b,0]),
                "pred_start": float(pred[ti,b,0]),
                "true_end": float(ans[ti,b,-1]),
                "pred_end": float(pred[ti,b,-1]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_target_batch.csv", index=False, float_format="%.8f")

    w = int(args.window)
    win_rows = []
    for ti, name in enumerate(TARGET_NAMES):
        for b in range(BATCHES):
            for s in range(0, PRED_LENGTH, w):
                e = min(PRED_LENGTH, s + w)
                ee = err[ti, b, s:e]
                win_rows.append({
                    "target": name,
                    "batch": b,
                    "start_h": s,
                    "end_h": e,
                    "mse": float(np.mean(ee ** 2)),
                    "rmse": float(np.sqrt(np.mean(ee ** 2))),
                    "bias": float(np.mean(ee)),
                    "true_mean": float(ans[ti,b,s:e].mean()),
                    "pred_mean": float(pred[ti,b,s:e].mean()),
                })
    wdf = pd.DataFrame(win_rows).sort_values("mse", ascending=False)
    wdf.to_csv(out_dir / "worst_windows.csv", index=False, float_format="%.8f")

    summary = {
        "submit": args.submit,
        "mse_f62": float(mse_tb[0].mean()),
        "mse_f63": float(mse_tb[1].mean()),
        "mse_f64": float(mse_tb[2].mean()),
        "mse_sum": float(mse_tb.mean(axis=1).sum()),
        "worst_target_batch": df.sort_values("mse", ascending=False).head(10).to_dict(orient="records"),
        "worst_windows_top10": wdf.head(10).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[SAVE] {out_dir}", flush=True)

if __name__ == "__main__":
    main()
