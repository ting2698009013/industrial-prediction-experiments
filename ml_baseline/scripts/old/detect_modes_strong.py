"""
Strong mode detection for Zhongkong Cup industrial time series.

Purpose
-------
Generate mode_labels.npy from cleaned train data using only input features.
It supports two complementary methods:

1) boundary method:
   Use known training-set mode boundaries from the existing preprocessing report.
   This is the most stable way to reproduce previous feature engineering.

2) classifier method:
   Train a classifier using the boundary labels as pseudo-labels, based on
   strong mode-separating input features. This model can later be applied to
   validation/test question.npy.

3) unsupervised method:
   Smooth strong mode columns, cluster into 3 modes, then enforce long segments.
   This is for diagnosis, not recommended as the first training label source.

No target columns 62/63/64 are used.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


# Human-visible mode-change columns from plots + audit top columns.
DEFAULT_MODE_COLS = [13, 28, 29, 33, 34, 36, 38, 61]
EXTENDED_MODE_COLS = [13, 28, 29, 33, 34, 36, 38, 61, 3, 60, 16, 31, 9, 19, 32]

# Existing report boundaries. Labels are kept consistent with the report:
# mode 2: [0,108037), mode 0: [108037,304397), mode 1: [304397,N)
DEFAULT_BOUNDARIES = [108037, 304397]
DEFAULT_SEG_LABELS = [2, 0, 1]


@dataclass
class Segment:
    start: int
    end: int
    label: int
    length: int


def rolling_mean_2d(x: np.ndarray, win: int) -> np.ndarray:
    """Centered-ish rolling mean with edge padding, memory safe for 600k x small F."""
    if win <= 1:
        return x.astype(np.float32, copy=True)
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    xp = np.pad(x.astype(np.float64), ((pad_left, pad_right), (0, 0)), mode="edge")
    cs = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(xp, axis=0)])
    sm = (cs[win:] - cs[:-win]) / float(win)
    return sm.astype(np.float32)


def make_boundary_labels(n: int, boundaries: list[int], labels: list[int]) -> np.ndarray:
    assert len(labels) == len(boundaries) + 1
    y = np.empty(n, dtype=np.int16)
    starts = [0] + boundaries
    ends = boundaries + [n]
    for s, e, lab in zip(starts, ends, labels):
        y[s:e] = int(lab)
    return y


def segments_from_labels(labels: np.ndarray) -> list[Segment]:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [Segment(int(s), int(e), int(labels[s]), int(e - s)) for s, e in zip(starts, ends)]


def majority_smooth(labels: np.ndarray, win: int) -> np.ndarray:
    """Fast majority smoothing for labels {0,1,2}."""
    if win <= 1:
        return labels.copy()
    labels = labels.astype(np.int16, copy=False)
    scores = []
    for m in range(3):
        scores.append(rolling_mean_2d((labels == m).astype(np.float32)[:, None], win)[:, 0])
    return np.argmax(np.vstack(scores), axis=0).astype(np.int16)


def merge_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Merge short islands into the longer adjacent segment."""
    out = labels.astype(np.int16, copy=True)
    changed = True
    while changed:
        changed = False
        segs = segments_from_labels(out)
        if len(segs) <= 1:
            break
        for k, seg in enumerate(segs):
            if seg.length >= min_len:
                continue
            if k == 0:
                new_lab = segs[k + 1].label
            elif k == len(segs) - 1:
                new_lab = segs[k - 1].label
            else:
                left, right = segs[k - 1], segs[k + 1]
                new_lab = left.label if left.length >= right.length else right.label
            out[seg.start:seg.end] = new_lab
            changed = True
            break
    return out


def build_mode_features(data: np.ndarray, cols: list[int], smooth_win: int) -> np.ndarray:
    raw = np.asarray(data[:, cols], dtype=np.float32)
    sm = rolling_mean_2d(raw, smooth_win)
    # Include both smoothed level and first-order slow difference; abrupt changes are useful.
    diff = np.zeros_like(sm)
    diff[1:] = sm[1:] - sm[:-1]
    return np.hstack([sm, diff]).astype(np.float32)


def fit_classifier(X_feat: np.ndarray, y: np.ndarray, method: str, stride: int, seed: int):
    idx = np.arange(0, len(y), max(1, stride))
    Xs, ys = X_feat[idx], y[idx]
    if method == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=200,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    elif method == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown classifier: {method}")
    pipe = Pipeline([("scaler", RobustScaler()), ("clf", clf)])
    pipe.fit(Xs, ys)
    pred = pipe.predict(X_feat)
    return pipe, pred.astype(np.int16)


def unsupervised_gmm(X_feat: np.ndarray, stride: int, seed: int) -> np.ndarray:
    idx = np.arange(0, X_feat.shape[0], max(1, stride))
    scaler = RobustScaler().fit(X_feat[idx])
    Xs = scaler.transform(X_feat[idx])
    gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=seed, reg_covar=1e-5)
    gmm.fit(Xs)
    lab = gmm.predict(scaler.transform(X_feat)).astype(np.int16)
    return lab


def save_labels_and_report(labels: np.ndarray, out_dir: str, name: str, extra: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}.npy"), labels.astype(np.int16))
    segs = segments_from_labels(labels)
    report = {
        "name": name,
        "n": int(len(labels)),
        "counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "n_changes": int(len(segs) - 1),
        "segments": [asdict(s) for s in segs],
    }
    if extra:
        report.update(extra)
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="clean train npy, shape=(600000,65)")
    ap.add_argument("--out", default="runs/modes_strong")
    ap.add_argument("--cols", default="extended", choices=["visible", "extended"])
    ap.add_argument("--smooth-win", type=int, default=1001)
    ap.add_argument("--min-segment-len", type=int, default=10000)
    ap.add_argument("--sample-stride", type=int, default=10)
    ap.add_argument("--classifier", default="rf", choices=["rf", "hgb"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.data, mmap_mode="r")
    n = data.shape[0]
    cols = DEFAULT_MODE_COLS if args.cols == "visible" else EXTENDED_MODE_COLS
    print(f"[LOAD] {args.data}: shape={data.shape}, mode_cols={cols}", flush=True)

    y_boundary = make_boundary_labels(n, DEFAULT_BOUNDARIES, DEFAULT_SEG_LABELS)
    boundary_report = save_labels_and_report(y_boundary, args.out, "mode_labels_known_boundaries", {
        "method": "fixed_report_boundaries",
        "boundaries": DEFAULT_BOUNDARIES,
        "segment_labels": DEFAULT_SEG_LABELS,
    })
    print("[SAVE] boundary labels", boundary_report["counts"], boundary_report["segments"], flush=True)

    X_feat = build_mode_features(data, cols, args.smooth_win)

    # Classifier trained on known segments, then predicted from features only.
    model, pred_cls = fit_classifier(X_feat, y_boundary, args.classifier, args.sample_stride, args.seed)
    pred_cls = majority_smooth(pred_cls, max(101, args.smooth_win // 2))
    pred_cls = merge_short_segments(pred_cls, args.min_segment_len)
    acc = float(accuracy_score(y_boundary, pred_cls))
    cm = confusion_matrix(y_boundary, pred_cls, labels=[0, 1, 2]).tolist()
    cls_report = save_labels_and_report(pred_cls, args.out, "mode_labels_classifier", {
        "method": f"{args.classifier}_trained_on_known_boundaries",
        "mode_cols": cols,
        "smooth_win": args.smooth_win,
        "min_segment_len": args.min_segment_len,
        "agreement_with_known_boundaries": acc,
        "confusion_matrix_labels_0_1_2": cm,
    })
    joblib.dump({
        "model": model,
        "cols": cols,
        "smooth_win": args.smooth_win,
        "min_segment_len": args.min_segment_len,
        "label_order": [0, 1, 2],
    }, os.path.join(args.out, "mode_classifier.joblib"))
    print("[SAVE] classifier labels", cls_report["counts"], "agree=", acc, flush=True)

    # Unsupervised diagnosis labels.
    pred_gmm = unsupervised_gmm(X_feat, args.sample_stride, args.seed)
    pred_gmm = majority_smooth(pred_gmm, max(101, args.smooth_win // 2))
    pred_gmm = merge_short_segments(pred_gmm, args.min_segment_len)
    gmm_report = save_labels_and_report(pred_gmm, args.out, "mode_labels_gmm_diagnostic", {
        "method": "unsupervised_gmm_diagnostic",
        "mode_cols": cols,
        "smooth_win": args.smooth_win,
        "min_segment_len": args.min_segment_len,
    })
    print("[SAVE] gmm diagnostic labels", gmm_report["counts"], flush=True)

    # Per-mode means/stds for sanity check.
    rows = []
    for source_name, labs in [
        ("known_boundaries", y_boundary),
        ("classifier", pred_cls),
        ("gmm_diagnostic", pred_gmm),
    ]:
        for c in cols:
            for m in [0, 1, 2]:
                vals = np.asarray(data[labs == m, c], dtype=np.float64)
                rows.append({
                    "source": source_name,
                    "feature": c,
                    "mode": m,
                    "count": int(vals.size),
                    "mean": float(vals.mean()) if vals.size else np.nan,
                    "std": float(vals.std()) if vals.size else np.nan,
                    "min": float(vals.min()) if vals.size else np.nan,
                    "max": float(vals.max()) if vals.size else np.nan,
                })
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "mode_feature_stats.csv"), index=False)

    # Default label for training: known boundaries, because this reproduces previous work exactly.
    np.save(os.path.join(args.out, "mode_labels.npy"), y_boundary.astype(np.int16))
    print(f"[DONE] default mode_labels.npy = known_boundaries. out={args.out}", flush=True)


if __name__ == "__main__":
    main()
