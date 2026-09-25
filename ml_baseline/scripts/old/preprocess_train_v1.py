"""
preprocess_train_v1.py

Only processes the training set for the industrial time-series prediction task.

Input:
    train.npy, expected shape: (600000, 65)

Outputs:
    cleaned_train_v1.npy
        Cleaned training data in competition-style batch format:
        shape = (65, 10, batch_length)

    y_targets_v1.npy
        Target signals after cleaning:
        shape = (3, 10, batch_length), corresponding to features 62, 63, 64

    X_<feature_set>_v1.npy
        Engineered input features:
        shape = (10, batch_length, n_engineered_features)

    feature_names_<feature_set>_v1.json
        Names of engineered feature columns, in the same order as X.

    preprocess_summary_v1.json
        Processing configuration and output shape summary.

Feature sets:
    base     : cleaned raw input features only
    diff     : base + diff1 + diff5
    lag      : base + lag1 + lag5 + lag10 + lag30
    rolling  : base + rolling_mean10 + rolling_mean30 + rolling_mean60
    all      : base + diff + lag + rolling

Notes:
    - This script does NOT split train/valid.
    - This script does NOT standardize features, because standardization should be
      fitted only inside the later model-training split to avoid leakage.
    - Current values of target columns 62, 63, 64 are NOT used in X.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

N_BATCHES = 10
N_FEATURES = 65
TARGET_COLS = [62, 63, 64]

# Zero-variance columns found by EDA report.
CONSTANT_COLS = [5, 6, 35, 39, 40, 41, 42, 43, 44, 45, 46]

# Only 0~61 are used as input features. Target cols 62~64 are labels only.
INPUT_COLS = [c for c in range(62) if c not in CONSTANT_COLS]

DIFF_STEPS = [1, 5]
LAG_STEPS = [1, 5, 10, 30]
ROLLING_WINDOWS = [10, 30, 60]


def load_train_as_batches(path: Path, n_batches: int = N_BATCHES) -> np.ndarray:
    """Load train.npy and reshape from (T, 65) to (65, 10, batch_length)."""
    data = np.load(path)

    if data.ndim != 2:
        raise ValueError(f"Training data should be 2D, got shape={data.shape}")
    if data.shape[1] != N_FEATURES:
        raise ValueError(f"Expected 65 features, got shape={data.shape}")

    seq_len, n_features = data.shape
    batch_len = seq_len // n_batches
    usable_len = batch_len * n_batches

    if usable_len != seq_len:
        print(f"[warn] sequence length {seq_len} is not divisible by {n_batches}; "
              f"trimming to {usable_len}")
        data = data[:usable_len]

    # (T, F) -> (batch, batch_len, F) -> (F, batch, batch_len)
    data_3d = data.reshape(n_batches, batch_len, n_features).transpose(2, 0, 1)
    return data_3d.astype(np.float32, copy=False)


def clean_one_batch(batch_2d: np.ndarray) -> pd.DataFrame:
    """Clean a single batch. Input shape: (batch_length, 65)."""
    df = pd.DataFrame(batch_2d, columns=list(range(N_FEATURES)))

    # Replace inf values first, then fill NaN in time order.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    # Fallback: fill remaining NaN with batch-wise median, then zero if a whole column is NaN.
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0.0)

    return df.astype(np.float32)


def clean_train_batches(data_3d: np.ndarray) -> np.ndarray:
    """Clean every batch independently. Return shape: (65, 10, batch_length)."""
    n_features, n_batches, batch_len = data_3d.shape
    cleaned = np.empty_like(data_3d, dtype=np.float32)

    for b in range(n_batches):
        # (65, L) -> (L, 65)
        df = clean_one_batch(data_3d[:, b, :].T)
        cleaned[:, b, :] = df.to_numpy(dtype=np.float32).T
        print(f"[clean] batch {b + 1}/{n_batches} done")

    return cleaned


def add_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feat = df[INPUT_COLS].copy()
    names = [f"f{c}" for c in INPUT_COLS]
    feat.columns = names
    return feat, names


def add_diff_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    names = []
    for col in INPUT_COLS:
        for step in DIFF_STEPS:
            name = f"f{col}_diff{step}"
            frames.append(df[col].diff(step).rename(name))
            names.append(name)
    return pd.concat(frames, axis=1), names


def add_lag_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    names = []
    for col in INPUT_COLS:
        for lag in LAG_STEPS:
            name = f"f{col}_lag{lag}"
            frames.append(df[col].shift(lag).rename(name))
            names.append(name)
    return pd.concat(frames, axis=1), names


def add_rolling_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    names = []
    for col in INPUT_COLS:
        for win in ROLLING_WINDOWS:
            name = f"f{col}_rollmean{win}"
            frames.append(df[col].rolling(win, min_periods=1).mean().rename(name))
            names.append(name)
    return pd.concat(frames, axis=1), names


def build_features_for_batch(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str]]:
    """Build selected feature set for one cleaned batch."""
    base, base_names = add_base_features(df)
    parts = [base]
    names = list(base_names)

    if feature_set in {"diff", "all"}:
        diff, diff_names = add_diff_features(df)
        parts.append(diff)
        names.extend(diff_names)

    if feature_set in {"lag", "all"}:
        lag, lag_names = add_lag_features(df)
        parts.append(lag)
        names.extend(lag_names)

    if feature_set in {"rolling", "all"}:
        rolling, rolling_names = add_rolling_features(df)
        parts.append(rolling)
        names.extend(rolling_names)

    if feature_set == "base":
        pass
    elif feature_set not in {"diff", "lag", "rolling", "all"}:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    feat = pd.concat(parts, axis=1)

    # Feature engineering creates NaN at the beginning because of diff/lag.
    # Fill only inside this batch.
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill().bfill().fillna(0.0)

    return feat.astype(np.float32), names


def build_feature_set(cleaned_3d: np.ndarray, feature_set: str) -> Tuple[np.ndarray, List[str]]:
    """Build features for all batches. Return shape: (10, batch_length, n_features)."""
    n_features, n_batches, batch_len = cleaned_3d.shape
    x_batches = []
    feature_names = None

    for b in range(n_batches):
        df = pd.DataFrame(cleaned_3d[:, b, :].T, columns=list(range(n_features)))
        feat, names = build_features_for_batch(df, feature_set)

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names are inconsistent across batches.")

        x_batches.append(feat.to_numpy(dtype=np.float32))
        print(f"[feature:{feature_set}] batch {b + 1}/{n_batches} done, shape={feat.shape}")

    x = np.stack(x_batches, axis=0).astype(np.float32, copy=False)
    return x, feature_names or []


def save_json(obj: Dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess training data only.")
    parser.add_argument("--input", type=Path, required=True, help="Path to train.npy")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--feature-set",
        choices=["base", "diff", "lag", "rolling", "all"],
        default="all",
        help="Which feature set to generate.",
    )
    parser.add_argument(
        "--save-cleaned",
        action="store_true",
        help="Also save cleaned_train_v1.npy. It is useful but takes disk space.",
    )
    parser.add_argument("--prefix", type=str, default="v1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.input}")
    raw_3d = load_train_as_batches(args.input)
    print(f"[load] batched shape: {raw_3d.shape}")

    print("[clean] start")
    cleaned_3d = clean_train_batches(raw_3d)

    y_targets = cleaned_3d[TARGET_COLS, :, :].astype(np.float32, copy=False)
    y_path = args.out_dir / f"y_targets_{args.prefix}.npy"
    np.save(y_path, y_targets)
    print(f"[save] {y_path}, shape={y_targets.shape}")

    if args.save_cleaned:
        clean_path = args.out_dir / f"cleaned_train_{args.prefix}.npy"
        np.save(clean_path, cleaned_3d)
        print(f"[save] {clean_path}, shape={cleaned_3d.shape}")

    print(f"[feature] build feature_set={args.feature_set}")
    x, feature_names = build_feature_set(cleaned_3d, args.feature_set)

    x_path = args.out_dir / f"X_{args.feature_set}_{args.prefix}.npy"
    names_path = args.out_dir / f"feature_names_{args.feature_set}_{args.prefix}.json"
    summary_path = args.out_dir / f"preprocess_summary_{args.prefix}.json"

    np.save(x_path, x)
    print(f"[save] {x_path}, shape={x.shape}")

    save_json({"feature_names": feature_names}, names_path)
    print(f"[save] {names_path}, n_features={len(feature_names)}")

    summary = {
        "input_file": str(args.input),
        "prefix": args.prefix,
        "raw_train_expected_shape": "(600000, 65)",
        "batched_shape": list(raw_3d.shape),
        "cleaning": [
            "reshape train to (65, 10, batch_length)",
            "clean each batch independently",
            "replace inf/-inf with NaN",
            "ffill, then bfill, then batch-wise median, then 0 fallback",
        ],
        "target_cols": TARGET_COLS,
        "constant_cols_removed_from_X": CONSTANT_COLS,
        "input_cols_used_for_X": INPUT_COLS,
        "feature_set": args.feature_set,
        "diff_steps": DIFF_STEPS if args.feature_set in {"diff", "all"} else [],
        "lag_steps": LAG_STEPS if args.feature_set in {"lag", "all"} else [],
        "rolling_mean_windows": ROLLING_WINDOWS if args.feature_set in {"rolling", "all"} else [],
        "outputs": {
            "X": {"path": str(x_path), "shape": list(x.shape)},
            "y_targets": {"path": str(y_path), "shape": list(y_targets.shape)},
            "feature_names": str(names_path),
        },
        "notes": [
            "No train/valid split is performed here.",
            "No standardization is performed here; fit scaler later inside model training split.",
            "Current target columns 62/63/64 are not included in X.",
        ],
    }
    if args.save_cleaned:
        summary["outputs"]["cleaned_train"] = {
            "path": str(args.out_dir / f"cleaned_train_{args.prefix}.npy"),
            "shape": list(cleaned_3d.shape),
        }

    save_json(summary, summary_path)
    print(f"[save] {summary_path}")
    print("[done]")


if __name__ == "__main__":
    main()
