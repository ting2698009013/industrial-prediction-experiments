"""
build_features_v1.py

只负责基于 cleaned_train_v1.npy 做特征工程，不负责清洗。

feature_set 可选:
    base    : 原始输入特征
    diff    : 原始输入特征 + diff1 + diff5
    lag     : 原始输入特征 + lag1 + lag5 + lag10 + lag30
    rolling : 原始输入特征 + rolling_mean10 + rolling_mean30 + rolling_mean60
    all     : 原始输入特征 + diff + lag + rolling
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLS = [62, 63, 64]
CONSTANT_COLS = [5, 6, 35, 39, 40, 41, 42, 43, 44, 45, 46]
INPUT_COLS = [i for i in range(62) if i not in CONSTANT_COLS]

DIFF_STEPS = [1, 5]
LAG_STEPS = [1, 5, 10, 30]
ROLLING_WINDOWS = [10, 30, 60]


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_text(prompt: str, default: str) -> str:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return value if value else default


def choose_feature_set() -> str:
    options = {"1": "base", "2": "diff", "3": "lag", "4": "rolling", "5": "all"}
    print("\n请选择要生成的特征工程版本:")
    print("1. base    = 原始输入特征")
    print("2. diff    = 原始输入特征 + diff1 + diff5")
    print("3. lag     = 原始输入特征 + lag1 + lag5 + lag10 + lag30")
    print("4. rolling = 原始输入特征 + rolling_mean10 + rolling_mean30 + rolling_mean60")
    print("5. all     = 原始输入特征 + diff + lag + rolling")
    choice = input("请输入 1/2/3/4/5，直接回车默认 all: ").strip()
    if not choice:
        return "all"
    if choice not in options:
        raise ValueError("无效选择，只能输入 1/2/3/4/5")
    return options[choice]


def build_features_for_batch(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    features = []
    feature_names = []

    base = df[INPUT_COLS].copy()
    base.columns = [f"feature_{c:02d}" for c in INPUT_COLS]
    features.append(base)
    feature_names.extend(base.columns.tolist())

    if feature_set in ["diff", "all"]:
        for step in DIFF_STEPS:
            diff = df[INPUT_COLS].diff(step)
            diff.columns = [f"feature_{c:02d}_diff{step}" for c in INPUT_COLS]
            features.append(diff)
            feature_names.extend(diff.columns.tolist())

    if feature_set in ["lag", "all"]:
        for step in LAG_STEPS:
            lag = df[INPUT_COLS].shift(step)
            lag.columns = [f"feature_{c:02d}_lag{step}" for c in INPUT_COLS]
            features.append(lag)
            feature_names.extend(lag.columns.tolist())

    if feature_set in ["rolling", "all"]:
        for window in ROLLING_WINDOWS:
            roll = df[INPUT_COLS].rolling(window=window, min_periods=1).mean()
            roll.columns = [f"feature_{c:02d}_rollmean{window}" for c in INPUT_COLS]
            features.append(roll)
            feature_names.extend(roll.columns.tolist())

    X = pd.concat(features, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0)
    return X.astype(np.float32), feature_names


def build_all_features(cleaned_3d: np.ndarray, feature_set: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_features, n_batches, _ = cleaned_3d.shape
    X_batches = []
    feature_names_ref = None

    for b in range(n_batches):
        batch = cleaned_3d[:, b, :].T
        df = pd.DataFrame(batch, columns=list(range(n_features)))
        X_df, feature_names = build_features_for_batch(df, feature_set)
        if feature_names_ref is None:
            feature_names_ref = feature_names
        elif feature_names != feature_names_ref:
            raise RuntimeError("不同 batch 生成的特征名不一致")
        X_batches.append(X_df.to_numpy(dtype=np.float32))
        print(f"[特征] batch {b}: X shape = {X_df.shape}")

    X = np.stack(X_batches, axis=0).astype(np.float32)
    y = cleaned_3d[TARGET_COLS, :, :].astype(np.float32)
    return X, y, feature_names_ref


def main():
    print("=" * 70)
    print("训练集特征工程脚本 build_features_v1.py")
    print("只做特征工程，不做清洗")
    print("=" * 70)

    cleaned_path = ask_path("请输入清洗后的训练集路径", "data/processed/cleaned_train_v1.npy")
    out_dir = ask_path("请输入输出目录", "data/processed")
    prefix = ask_text("请输入版本前缀", "v1")
    feature_set = choose_feature_set()

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[读取] {cleaned_path}")
    cleaned = np.load(cleaned_path)
    if cleaned.ndim != 3 or cleaned.shape[0] != 65:
        raise ValueError(f"cleaned_train 应为 shape=(65, 10, batch_len)，但得到 {cleaned.shape}")

    print(f"[形状] cleaned_train: {cleaned.shape}")
    print(f"[版本] feature_set = {feature_set}")

    X, y, feature_names = build_all_features(cleaned, feature_set)

    X_path = out_dir / f"X_{feature_set}_{prefix}.npy"
    y_path = out_dir / f"y_targets_{prefix}.npy"
    names_path = out_dir / f"feature_names_{feature_set}_{prefix}.json"
    summary_path = out_dir / f"feature_summary_{feature_set}_{prefix}.json"

    np.save(X_path, X)
    np.save(y_path, y)
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    summary = {
        "script": "build_features_v1.py",
        "version_prefix": prefix,
        "feature_set": feature_set,
        "cleaned_input_path": str(cleaned_path),
        "X_output_path": str(X_path),
        "y_output_path": str(y_path),
        "feature_names_path": str(names_path),
        "cleaned_shape": list(cleaned.shape),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "target_cols": TARGET_COLS,
        "constant_cols_removed_from_X": CONSTANT_COLS,
        "input_cols_used": INPUT_COLS,
        "diff_steps": DIFF_STEPS if feature_set in ["diff", "all"] else [],
        "lag_steps": LAG_STEPS if feature_set in ["lag", "all"] else [],
        "rolling_windows": ROLLING_WINDOWS if feature_set in ["rolling", "all"] else [],
        "note": "X uses only feature_00~feature_61 excluding constant columns. Current feature_62~64 are not used as input.",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"X 特征文件: {X_path}")
    print(f"y 目标文件: {y_path}")
    print(f"特征名文件: {names_path}")
    print(f"说明文件: {summary_path}")


if __name__ == "__main__":
    main()
