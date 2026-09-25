"""
clean_train_v2.py

用途：
    基于原始 train.npy 生成 cleaned_train_v2.npy。
    v2 是“保守局部毛刺修正版”。

重要原则：
    1. 从原始 train.npy 重新开始，不在 v1 基础上继续清洗
    2. 先做 v1 的 NaN / Inf 清洗
    3. 再读取局部异常检测结果，只修正很保守的一小部分点
    4. 检测到的异常点 != 实际清洗的点
    5. 目标列 62,63,64 不清洗
    6. 常量列、近常量列、低唯一值列、feature_61 默认不清洗

输入：
    data/raw/train.npy
    reports/local_outliers_v2/local_outlier_summary.csv
    reports/local_outliers_v2/local_outlier_runs.csv
    reports/local_outliers_v2/feature_profile.csv

输出：
    data/processed/cleaned_train_v2.npy
    data/processed/clean_v2_modified_mask.npy
    data/processed/clean_v2_summary.csv
    data/processed/clean_v2_config.json

运行：
    python .\\scripts\\clean_train_v2.py
"""

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd


N_BATCHES = 10
N_FEATURES = 65

TARGET_COLS = {62, 63, 64}

# 根据前期分析手动固定跳过，避免小数据/阈值变化导致结果复现不稳定
# 5,6,39~46 是严格常量列；35 是近常量列；61 是量化/小范围离散波动列，不做中位数替换
MANUAL_SKIP_COLS = {
    5, 6, 35,
    39, 40, 41, 42, 43, 44, 45, 46,
    61,
}


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_int(prompt: str, default: int) -> int:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return int(value) if value else default


def ask_float(prompt: str, default: float) -> float:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return float(value) if value else default


def ask_yes_no(prompt: str, default: bool) -> bool:
    default_text = "Y" if default else "N"
    value = input(f"{prompt}\n默认: {default_text}，输入 y/n，直接回车使用默认值: ").strip().lower()
    if not value:
        return default
    return value in ["y", "yes", "1", "true", "是"]


def to_batch_format(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        seq_len, n_features = data.shape
        if n_features != N_FEATURES:
            raise ValueError(f"二维数据应为 (*, 65)，但得到 {data.shape}")
        batch_len = seq_len // N_BATCHES
        used_len = batch_len * N_BATCHES
        if used_len != seq_len:
            print(f"[提示] 原始长度 {seq_len} 不能被 {N_BATCHES} 整除，将截断到 {used_len}")
        data = data[:used_len]
        data = data.reshape(N_BATCHES, batch_len, n_features)
        data = data.transpose(2, 0, 1)
        return data

    if data.ndim == 3:
        if data.shape[0] != N_FEATURES:
            raise ValueError(f"三维数据应为 (65, 10, batch_len)，但得到 {data.shape}")
        return data

    raise ValueError(f"不支持的数据形状: {data.shape}")


def clean_nan_inf_one_series(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    s = pd.Series(x.astype(np.float32))
    n_nan_before = int(s.isna().sum())
    n_inf_before = int(np.isinf(s.to_numpy()).sum())

    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.ffill().bfill()

    if s.isna().any():
        median = s.median()
        if pd.isna(median):
            median = 0.0
        s = s.fillna(median)

    return s.to_numpy(dtype=np.float32), n_nan_before, n_inf_before


def basic_clean(data_3d: np.ndarray) -> tuple[np.ndarray, dict]:
    cleaned = data_3d.astype(np.float32).copy()
    total_nan = 0
    total_inf = 0

    for feature in range(cleaned.shape[0]):
        for batch in range(cleaned.shape[1]):
            fixed, n_nan, n_inf = clean_nan_inf_one_series(cleaned[feature, batch, :])
            cleaned[feature, batch, :] = fixed
            total_nan += n_nan
            total_inf += n_inf

    info = {
        "nan_before": total_nan,
        "inf_before": total_inf,
        "nan_after": int(np.isnan(cleaned).sum()),
        "inf_after": int(np.isinf(cleaned).sum()),
    }
    return cleaned, info


def load_csv_dicts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_summary_map(summary_path: Path) -> dict[tuple[int, int], dict]:
    rows = load_csv_dicts(summary_path)
    out = {}
    for row in rows:
        feature = int(row["feature"])
        batch = int(row["batch"])
        out[(feature, batch)] = row
    return out


def load_profile_map(profile_path: Path) -> dict[int, dict]:
    rows = load_csv_dicts(profile_path)
    out = {}
    for row in rows:
        feature = int(row["feature"])
        out[feature] = row
    return out


def get_bool_from_csv(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ["true", "1", "yes", "y"]


def rolling_median_values(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x.astype(float))
    med = s.rolling(
        window=window,
        center=True,
        min_periods=max(5, window // 5),
    ).median()

    med = med.ffill().bfill()

    if med.isna().any():
        global_median = float(np.nanmedian(x))
        if not np.isfinite(global_median):
            global_median = 0.0
        med = med.fillna(global_median)

    return med.to_numpy(dtype=np.float32)


def should_skip_feature(
    feature: int,
    profile: dict | None,
    skip_low_unique: bool,
    low_unique_threshold: int,
    skip_near_zero_by_profile: bool,
) -> tuple[bool, str]:
    if feature in TARGET_COLS:
        return True, "target_col"

    if feature in MANUAL_SKIP_COLS:
        return True, "manual_skip"

    if profile is None:
        return False, ""

    if get_bool_from_csv(profile, "is_zero_var"):
        return True, "zero_variance"

    if skip_near_zero_by_profile and get_bool_from_csv(profile, "is_near_zero_var"):
        return True, "near_zero_variance"

    if skip_low_unique:
        try:
            n_unique = int(float(profile.get("n_unique", "999999")))
        except ValueError:
            n_unique = 999999
        if n_unique <= low_unique_threshold:
            return True, f"low_unique_le_{low_unique_threshold}"

    return False, ""


def main():
    print("=" * 72)
    print("clean_train_v2：保守局部毛刺清洗")
    print("=" * 72)

    raw_path = ask_path("请输入原始 train.npy 路径", "data/raw/train.npy")
    summary_path = ask_path(
        "请输入 local_outlier_summary.csv 路径",
        "reports/local_outliers_v2/local_outlier_summary.csv",
    )
    runs_path = ask_path(
        "请输入 local_outlier_runs.csv 路径",
        "reports/local_outliers_v2/local_outlier_runs.csv",
    )
    profile_path = ask_path(
        "请输入 feature_profile.csv 路径",
        "reports/local_outliers_v2/feature_profile.csv",
    )
    out_dir = ask_path("请输入输出目录", "data/processed")

    prefix = input("请输入版本前缀，直接回车使用 v2: ").strip() or "v2"

    max_run_len = ask_int("只清洗连续异常段长度 <= 该值的点，建议 3", 3)
    max_feature_batch_ratio = ask_float("如果某 feature-batch 异常比例超过该值则跳过，建议 0.01", 0.01)
    rolling_window = ask_int("替换用 rolling median 窗口，建议 201", 201)

    skip_low_unique = ask_yes_no("是否自动跳过低唯一值列", True)
    low_unique_threshold = ask_int("低唯一值阈值，n_unique <= 该值跳过，建议 10", 10)

    skip_near_zero_by_profile = ask_yes_no(
        "是否根据 feature_profile.csv 的 is_near_zero_var 跳过近零方差列",
        True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[读取数据]")
    raw = to_batch_format(np.load(raw_path))
    print(f"raw shape = {raw.shape}")

    print("\n[基础清洗 NaN/Inf]")
    cleaned, basic_info = basic_clean(raw)
    print(basic_info)

    print("\n[读取异常检测结果]")
    summary_map = load_summary_map(summary_path)
    profile_map = load_profile_map(profile_path)
    run_rows = load_csv_dicts(runs_path)

    modified_mask = np.zeros_like(cleaned, dtype=bool)

    clean_summary_rows = []
    total_runs_seen = 0
    total_runs_modified = 0
    total_points_modified = 0

    feature_batch_decisions = {}

    for feature in range(N_FEATURES):
        profile = profile_map.get(feature)
        skip_feature, feature_skip_reason = should_skip_feature(
            feature,
            profile,
            skip_low_unique=skip_low_unique,
            low_unique_threshold=low_unique_threshold,
            skip_near_zero_by_profile=skip_near_zero_by_profile,
        )

        for batch in range(N_BATCHES):
            summary = summary_map.get((feature, batch), {})
            ratio = float(summary.get("outlier_ratio", 0.0) or 0.0)

            skip_batch = False
            batch_skip_reason = ""

            if skip_feature:
                skip_batch = True
                batch_skip_reason = feature_skip_reason
            elif ratio > max_feature_batch_ratio:
                skip_batch = True
                batch_skip_reason = f"outlier_ratio_gt_{max_feature_batch_ratio}"

            feature_batch_decisions[(feature, batch)] = {
                "skip": skip_batch,
                "reason": batch_skip_reason,
                "outlier_ratio": ratio,
                "modified_count": 0,
                "modified_runs": 0,
            }

    rolling_median_cache = {}

    for row in run_rows:
        if not row:
            continue

        feature = int(row["feature"])
        batch = int(row["batch"])
        start = int(row["start"])
        end = int(row["end"])
        length = int(row["length"])

        total_runs_seen += 1

        decision = feature_batch_decisions[(feature, batch)]

        if decision["skip"]:
            continue

        if length > max_run_len:
            continue

        key = (feature, batch)
        if key not in rolling_median_cache:
            rolling_median_cache[key] = rolling_median_values(
                cleaned[feature, batch, :],
                window=rolling_window,
            )

        med = rolling_median_cache[key]
        cleaned[feature, batch, start:end + 1] = med[start:end + 1]
        modified_mask[feature, batch, start:end + 1] = True

        decision["modified_count"] += length
        decision["modified_runs"] += 1
        total_points_modified += length
        total_runs_modified += 1

    for feature in range(N_FEATURES):
        profile = profile_map.get(feature, {})
        for batch in range(N_BATCHES):
            decision = feature_batch_decisions[(feature, batch)]
            summary = summary_map.get((feature, batch), {})

            clean_summary_rows.append({
                "feature": feature,
                "batch": batch,
                "skip": decision["skip"],
                "skip_reason": decision["reason"],
                "outlier_ratio": decision["outlier_ratio"],
                "detected_outlier_count": int(float(summary.get("outlier_count", 0) or 0)),
                "detected_run_count": int(float(summary.get("run_count", 0) or 0)),
                "modified_count": decision["modified_count"],
                "modified_ratio": decision["modified_count"] / raw.shape[2],
                "modified_runs": decision["modified_runs"],
                "n_unique": profile.get("n_unique", ""),
                "variance": profile.get("variance", ""),
                "std": profile.get("std", ""),
            })

    cleaned_path = out_dir / f"cleaned_train_{prefix}.npy"
    mask_path = out_dir / f"clean_{prefix}_modified_mask.npy"
    summary_out_path = out_dir / f"clean_{prefix}_summary.csv"
    config_path = out_dir / f"clean_{prefix}_config.json"

    np.save(cleaned_path, cleaned)
    np.save(mask_path, modified_mask)

    with open(summary_out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clean_summary_rows)

    config = {
        "script": "clean_train_v2.py",
        "raw_path": str(raw_path),
        "raw_shape_after_batch": list(raw.shape),
        "summary_path": str(summary_path),
        "runs_path": str(runs_path),
        "profile_path": str(profile_path),
        "prefix": prefix,
        "basic_clean": basic_info,
        "manual_skip_cols": sorted(MANUAL_SKIP_COLS),
        "target_cols": sorted(TARGET_COLS),
        "max_run_len": max_run_len,
        "max_feature_batch_ratio": max_feature_batch_ratio,
        "rolling_window": rolling_window,
        "skip_low_unique": skip_low_unique,
        "low_unique_threshold": low_unique_threshold,
        "skip_near_zero_by_profile": skip_near_zero_by_profile,
        "total_runs_seen": total_runs_seen,
        "total_runs_modified": total_runs_modified,
        "total_points_modified": total_points_modified,
        "total_points": int(cleaned.size),
        "modified_ratio_all_data": total_points_modified / int(cleaned.size),
        "outputs": {
            "cleaned_train": str(cleaned_path),
            "modified_mask": str(mask_path),
            "summary": str(summary_out_path),
            "config": str(config_path),
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"清洗后数据: {cleaned_path}")
    print(f"实际修改 mask: {mask_path}")
    print(f"清洗统计: {summary_out_path}")
    print(f"配置记录: {config_path}")
    print(f"实际修改点: {total_points_modified}")
    print(f"占全部数据比例: {total_points_modified / int(cleaned.size):.6%}")


if __name__ == "__main__":
    main()
