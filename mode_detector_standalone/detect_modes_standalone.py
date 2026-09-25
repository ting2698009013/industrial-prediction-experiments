"""
Standalone operating-mode detector for industrial time-series .npy files.

Input formats supported:
  1) train-like data:    shape = (N, features), usually (600000, 65)
  2) question-like data: shape = (features, batch, batch_len), usually (65, 10, 60000)

What it does:
  - uses only non-target feature columns by default;
  - fits a GMM on a sampled subset of the given dataset;
  - predicts mode labels for the full sequence;
  - smooths labels by majority voting;
  - outputs labels, mode segments, change points, diagnostics, and the fitted model.

Dependencies:
  numpy pandas scikit-learn joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler


# Candidate columns found useful in the previous project. They intentionally exclude targets 62/63/64.
DEFAULT_MODE_COLS = np.array([13, 28, 29, 33, 34, 36, 38, 61, 3, 60, 16, 31, 9, 19, 32], dtype=np.int32)
TARGET_COLS = {62, 63, 64}


def ensure_odd(x: int) -> int:
    x = int(max(1, x))
    return x if x % 2 == 1 else x + 1


def default_smooth_win(n: int, batch_len: int | None = None) -> int:
    """Auto smoothing window.

    For official length 600000 and batch_len 60000, this gives 1001.
    For smaller debugging datasets, it shrinks automatically.
    """
    n = int(n)
    if batch_len is None:
        batch_len = max(1, n // 10) if n >= 10 else max(1, n)
    win = min(2001, max(501, int(round(batch_len / 60.0))))
    if n < 10000:
        win = min(win, max(1, n // 5))
    win = min(win, n)
    return ensure_odd(win)


def smooth_labels_fast(labels: np.ndarray, win: int, n_modes: int) -> np.ndarray:
    """Majority smoothing with cumulative sums, avoiding slow per-row window loops."""
    labels = np.asarray(labels, dtype=np.int16)
    if len(labels) == 0 or win <= 1:
        return labels.copy()
    win = ensure_odd(min(int(win), len(labels)))
    half = win // 2
    padded = np.pad(labels, (half, half), mode="edge")
    counts = np.empty((len(labels), n_modes), dtype=np.int32)
    for m in range(n_modes):
        v = (padded == m).astype(np.int32)
        cs = np.concatenate([[0], np.cumsum(v)])
        counts[:, m] = cs[win:] - cs[:-win]
    return counts.argmax(axis=1).astype(np.int16)


def load_npy_as_sequence(path: str | Path, mmap: bool = True) -> tuple[np.ndarray, dict]:
    """Return a sequence view/copy with shape (N, features) plus metadata."""
    arr = np.load(path, mmap_mode="r" if mmap else None)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError(f"2-D input must be (N, features), got {arr.shape}")
        meta = {
            "kind": "train_like",
            "original_shape": list(arr.shape),
            "n": int(arr.shape[0]),
            "features": int(arr.shape[1]),
        }
        return arr, meta
    if arr.ndim == 3:
        # Official format: (features, batch, batch_len)
        if arr.shape[0] < 2:
            raise ValueError(f"3-D input should be (features, batch, batch_len), got {arr.shape}")
        seq = arr.transpose(1, 2, 0).reshape(arr.shape[1] * arr.shape[2], arr.shape[0])
        meta = {
            "kind": "question_like",
            "original_shape": list(arr.shape),
            "features": int(arr.shape[0]),
            "batches": int(arr.shape[1]),
            "batch_len": int(arr.shape[2]),
            "n": int(arr.shape[1] * arr.shape[2]),
        }
        return seq, meta
    raise ValueError(f"expected (N,features) or (features,batch,batch_len), got {arr.shape}")


def parse_feature_cols(spec: str, n_features: int) -> np.ndarray:
    """Parse feature columns. 'auto' uses DEFAULT_MODE_COLS that exist in the input."""
    if spec == "auto":
        cols = DEFAULT_MODE_COLS[DEFAULT_MODE_COLS < n_features]
    elif spec == "all_non_target":
        cols = np.array([i for i in range(n_features) if i not in TARGET_COLS], dtype=np.int32)
    else:
        cols = np.array([int(x.strip()) for x in spec.split(",") if x.strip() != ""], dtype=np.int32)
        bad = [int(c) for c in cols if c < 0 or c >= n_features]
        if bad:
            raise ValueError(f"feature ids out of range for {n_features} features: {bad}")
    cols = np.array([int(c) for c in cols if int(c) not in TARGET_COLS], dtype=np.int32)
    if len(cols) == 0:
        raise ValueError("no usable mode feature columns; targets 62/63/64 are excluded")
    return cols


def drop_near_constant_cols(data: np.ndarray, cols: np.ndarray, sample_stride: int) -> tuple[np.ndarray, list[int]]:
    sample = np.asarray(data[::sample_stride][:, cols], dtype=np.float32)
    std = sample.std(axis=0)
    keep = std > 1e-8
    kept = cols[keep]
    dropped = [int(c) for c in cols[~keep]]
    if len(kept) == 0:
        raise ValueError("all selected feature columns are near-constant on sampled data")
    return kept.astype(np.int32), dropped


def remap_by_first_occurrence(labels: np.ndarray, n_modes: int) -> tuple[np.ndarray, dict]:
    """Make labels easier to read: first appearing component becomes 0, next becomes 1, etc."""
    order = []
    for v in labels:
        iv = int(v)
        if iv not in order:
            order.append(iv)
        if len(order) == n_modes:
            break
    for iv in range(n_modes):
        if iv not in order:
            order.append(iv)
    mapping = {old: new for new, old in enumerate(order)}
    out = np.empty_like(labels, dtype=np.int16)
    for old, new in mapping.items():
        out[labels == old] = int(new)
    return out, mapping


def labels_to_segments(labels: np.ndarray, meta: dict) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int16)
    rows = []
    if len(labels) == 0:
        return pd.DataFrame(columns=["start", "end", "label", "length"])
    start = 0
    cur = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            rows.append(_segment_row(start, i, cur, meta))
            start = i
            cur = int(labels[i])
    rows.append(_segment_row(start, len(labels), cur, meta))
    return pd.DataFrame(rows)


def _segment_row(start: int, end: int, label: int, meta: dict) -> dict:
    row = {"start": int(start), "end": int(end), "label": int(label), "length": int(end - start)}
    if meta.get("kind") == "question_like":
        bl = int(meta["batch_len"])
        row.update({
            "start_batch": int(start // bl),
            "start_t_in_batch": int(start % bl),
            "end_batch": int((end - 1) // bl),
            "end_t_in_batch_exclusive": int(end % bl),
        })
    return row


def change_points_frame(labels: np.ndarray, meta: dict) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int16)
    changes = np.where(labels[1:] != labels[:-1])[0] + 1 if len(labels) > 1 else np.array([], dtype=np.int64)
    rows = []
    for pos in changes:
        row = {
            "pos": int(pos),
            "from_label": int(labels[pos - 1]),
            "to_label": int(labels[pos]),
        }
        if meta.get("kind") == "question_like":
            bl = int(meta["batch_len"])
            row.update({"batch": int(pos // bl), "t_in_batch": int(pos % bl)})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_separation_frame(data: np.ndarray, labels: np.ndarray, cols: np.ndarray, n_modes: int) -> pd.DataFrame:
    rows = []
    labels = np.asarray(labels, dtype=np.int16)
    for c in cols:
        x = np.asarray(data[:, int(c)], dtype=np.float64)
        means, stds, counts = [], [], []
        for m in range(n_modes):
            mask = labels == m
            counts.append(int(mask.sum()))
            if mask.sum() == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(x[mask].mean()))
                stds.append(float(x[mask].std() + 1e-12))
        valid_means = [v for v in means if np.isfinite(v)]
        valid_stds = [v for v in stds if np.isfinite(v)]
        score = float((max(valid_means) - min(valid_means)) / (np.mean(valid_stds) + 1e-12)) if valid_means else np.nan
        rows.append({"feature": int(c), "sep_score": score, "counts": str(counts), "means": str(means), "stds": str(stds)})
    return pd.DataFrame(rows).sort_values("sep_score", ascending=False)


def save_json(obj: dict, path: str | Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def detect_modes(
    input_path: str | Path,
    out_dir: str | Path,
    features: str = "auto",
    n_modes: int = 3,
    sample_stride: int = 10,
    smooth_win: str | int = "auto",
    scaler_type: str = "robust",
    covariance_type: str = "full",
    random_state: int = 42,
    n_init: int = 10,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_npy_as_sequence(input_path, mmap=True)
    n_features = int(meta["features"])
    cols = parse_feature_cols(features, n_features)
    cols, dropped_cols = drop_near_constant_cols(data, cols, max(1, int(sample_stride)))

    sample = np.asarray(data[::sample_stride][:, cols], dtype=np.float32)
    scaler = RobustScaler().fit(sample) if scaler_type == "robust" else StandardScaler().fit(sample)
    sample_scaled = scaler.transform(sample)

    gmm = GaussianMixture(
        n_components=int(n_modes),
        covariance_type=covariance_type,
        random_state=int(random_state),
        n_init=int(n_init),
        max_iter=500,
    )
    gmm.fit(sample_scaled)

    X_all = np.asarray(data[:, cols], dtype=np.float32)
    raw_labels = gmm.predict(scaler.transform(X_all)).astype(np.int16)

    batch_len = meta.get("batch_len")
    if smooth_win == "auto":
        win = default_smooth_win(len(raw_labels), batch_len=batch_len)
    else:
        win = ensure_odd(int(smooth_win))
    smoothed_raw = smooth_labels_fast(raw_labels, win, n_modes=int(n_modes))
    labels, mapping = remap_by_first_occurrence(smoothed_raw, n_modes=int(n_modes))

    # Core outputs
    np.save(out_dir / "mode_labels.npy", labels.astype(np.int16))
    np.save(out_dir / "mode_raw_labels.npy", raw_labels.astype(np.int16))
    if meta["kind"] == "question_like":
        batched = labels.reshape(int(meta["batches"]), int(meta["batch_len"]))
        np.save(out_dir / "mode_labels_batched.npy", batched.astype(np.int16))

    segments_df = labels_to_segments(labels, meta)
    changes_df = change_points_frame(labels, meta)
    sep_df = feature_separation_frame(data, labels, cols, n_modes=int(n_modes))

    segments_df.to_csv(out_dir / "mode_segments.csv", index=False)
    changes_df.to_csv(out_dir / "mode_change_points.csv", index=False)
    sep_df.to_csv(out_dir / "mode_feature_separation.csv", index=False)

    model_pack = {
        "kind": "standalone_mode_gmm",
        "version": 1,
        "cols": cols.astype(np.int32),
        "scaler": scaler,
        "gmm": gmm,
        "mapping_first_occurrence": mapping,
        "n_modes": int(n_modes),
        "sample_stride": int(sample_stride),
        "smooth_win": int(win),
        "scaler_type": scaler_type,
        "covariance_type": covariance_type,
        "input_meta": meta,
    }
    joblib.dump(model_pack, out_dir / "mode_model.joblib")

    counts = {int(i): int((labels == i).sum()) for i in range(int(n_modes))}
    changes = changes_df["pos"].tolist() if "pos" in changes_df.columns else []
    summary = {
        "input_path": str(input_path),
        "input_meta": meta,
        "n": int(len(labels)),
        "n_modes": int(n_modes),
        "counts": counts,
        "n_changes": int(len(changes)),
        "first_changes": [int(x) for x in changes[:50]],
        "smooth_win": int(win),
        "sample_stride": int(sample_stride),
        "features_used": [int(c) for c in cols],
        "features_dropped_near_constant": dropped_cols,
        "target_cols_excluded": sorted([c for c in TARGET_COLS if c < n_features]),
        "label_remap_raw_to_output": {int(k): int(v) for k, v in mapping.items()},
        "files": [
            "mode_labels.npy",
            "mode_raw_labels.npy",
            "mode_segments.csv",
            "mode_change_points.csv",
            "mode_feature_separation.csv",
            "mode_model.joblib",
            "summary.json",
        ],
    }
    if meta["kind"] == "question_like":
        summary["files"].append("mode_labels_batched.npy")

    save_json(summary, out_dir / "summary.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone GMM operating-mode detector for .npy time-series data")
    p.add_argument("--input", required=True, help="input .npy: (N,features) or (features,batch,batch_len)")
    p.add_argument("--out", default="runs/modes_detected", help="output directory")
    p.add_argument("--features", default="auto", help="auto, all_non_target, or comma-separated feature ids, e.g. 13,28,29,33")
    p.add_argument("--n-modes", type=int, default=3, help="number of operating modes; default 3")
    p.add_argument("--sample-stride", type=int, default=10, help="fit GMM on every Nth row; default 10")
    p.add_argument("--smooth-win", default="auto", help="auto or odd integer, e.g. 1001 or 2001")
    p.add_argument("--scaler", default="robust", choices=["robust", "standard"])
    p.add_argument("--covariance", default="full", choices=["full", "tied", "diag", "spherical"])
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    summary = detect_modes(
        input_path=args.input,
        out_dir=args.out,
        features=args.features,
        n_modes=args.n_modes,
        sample_stride=args.sample_stride,
        smooth_win=args.smooth_win,
        scaler_type=args.scaler,
        covariance_type=args.covariance,
        random_state=args.random_state,
        n_init=args.n_init,
    )

    print("[OK] mode detection finished")
    print(" output dir:", args.out)
    print(" labels    :", Path(args.out) / "mode_labels.npy")
    if summary["input_meta"]["kind"] == "question_like":
        print(" batched   :", Path(args.out) / "mode_labels_batched.npy")
    print(" segments  :", Path(args.out) / "mode_segments.csv")
    print(" changes   :", Path(args.out) / "mode_change_points.csv")
    print(" summary   :", Path(args.out) / "summary.json")
    print(json.dumps({
        "counts": summary["counts"],
        "n_changes": summary["n_changes"],
        "first_changes": summary["first_changes"][:20],
        "smooth_win": summary["smooth_win"],
        "features_used": summary["features_used"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
