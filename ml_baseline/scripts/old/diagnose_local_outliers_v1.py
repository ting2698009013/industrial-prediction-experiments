"""
diagnose_local_outliers_v1.py

用途:
    做局部异常诊断，不修改数据。
    使用 rolling median + rolling MAD 检测局部尖峰。
    输出异常掩码、统计表、连续异常段表。

为什么单独写诊断脚本:
    先判断异常值是孤立尖峰还是连续工况段。
    不建议直接在清洗脚本里盲目改数据。

输入:
    data/raw/train.npy 或 data/processed/cleaned_train_v1.npy
    shape 可以是 (600000, 65) 或 (65, 10, batch_len)

输出:
    reports/local_outliers_v1/
        local_outlier_summary.csv
        local_outlier_runs.csv
        local_outlier_mask.npy

运行:
    python .\scripts\diagnose_local_outliers_v1.py
"""

from pathlib import Path
import csv

import numpy as np
import pandas as pd


N_BATCHES = 10
N_FEATURES = 65


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_int(prompt: str, default: int) -> int:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return int(value) if value else default


def ask_float(prompt: str, default: float) -> float:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return float(value) if value else default


def ask_feature_list() -> list[int]:
    text = input(
        "请输入要诊断的特征列，例如 7,9,21,28,58,64；直接回车表示全部 0~64:\n"
    ).strip()
    if not text:
        return list(range(N_FEATURES))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def to_batch_format(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        seq_len, n_features = data.shape
        if n_features != N_FEATURES:
            raise ValueError(f"二维数据应为 (*, 65)，但得到 {data.shape}")
        batch_len = seq_len // N_BATCHES
        used_len = batch_len * N_BATCHES
        data = data[:used_len]
        data = data.reshape(N_BATCHES, batch_len, n_features)
        data = data.transpose(2, 0, 1)
        return data

    if data.ndim == 3:
        if data.shape[0] != N_FEATURES:
            raise ValueError(f"三维数据应为 (65, 10, batch_len)，但得到 {data.shape}")
        return data

    raise ValueError(f"不支持的数据形状: {data.shape}")


def rolling_mad_outlier_mask(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """
    局部异常检测:
        median = rolling median
        mad = rolling median(abs(x - median))
        score = abs(x - median) / (1.4826 * mad)
    """
    s = pd.Series(x.astype(float))
    med = s.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()

    global_scale = np.nanmedian(np.abs(x - np.nanmedian(x)))
    fallback = global_scale if np.isfinite(global_scale) and global_scale > 1e-8 else 1e-6
    mad = mad.fillna(fallback)
    mad = mad.mask(mad < 1e-8, fallback)

    score = abs_dev / (1.4826 * mad)
    # pandas 在部分环境里会返回只读 ndarray，不能使用 mask |= 这种原地操作
    mask = score.gt(threshold).fillna(False).to_numpy(dtype=bool).copy()

    # 原始 NaN / Inf 也标记为异常；这里用新数组赋值，避免只读数组报错
    mask = mask | (~np.isfinite(x))
    return mask


def find_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    runs = []
    start = idx[0]
    prev = idx[0]
    for current in idx[1:]:
        if current == prev + 1:
            prev = current
        else:
            runs.append((int(start), int(prev), int(prev - start + 1)))
            start = current
            prev = current
    runs.append((int(start), int(prev), int(prev - start + 1)))
    return runs


def main():
    print("=" * 72)
    print("局部异常诊断：rolling median + MAD")
    print("只检测，不清洗")
    print("=" * 72)

    input_path = ask_path("请输入要诊断的数据路径", "data/raw/train.npy")
    out_dir = ask_path("请输入输出目录", "reports/local_outliers_v1")
    window = ask_int("rolling 窗口大小，建议 101 或 201", 201)
    threshold = ask_float("异常阈值，建议 6~8，越大越保守", 8.0)
    features = ask_feature_list()

    out_dir.mkdir(parents=True, exist_ok=True)

    data = to_batch_format(np.load(input_path))
    n_features, n_batches, batch_len = data.shape
    mask_all = np.zeros_like(data, dtype=bool)

    summary_rows = []
    run_rows = []

    for feature in features:
        for batch in range(n_batches):
            x = data[feature, batch, :]
            mask = rolling_mad_outlier_mask(x, window=window, threshold=threshold)
            mask_all[feature, batch, :] = mask

            runs = find_runs(mask)
            lengths = [r[2] for r in runs]

            summary_rows.append({
                "feature": feature,
                "batch": batch,
                "outlier_count": int(mask.sum()),
                "outlier_ratio": float(mask.mean()),
                "run_count": len(runs),
                "max_run_length": int(max(lengths)) if lengths else 0,
                "short_run_1_3": int(sum(1 for l in lengths if 1 <= l <= 3)),
                "medium_run_4_20": int(sum(1 for l in lengths if 4 <= l <= 20)),
                "long_run_gt20": int(sum(1 for l in lengths if l > 20)),
            })

            for start, end, length in runs:
                run_rows.append({
                    "feature": feature,
                    "batch": batch,
                    "start": start,
                    "end": end,
                    "length": length,
                    "start_value": float(x[start]) if np.isfinite(x[start]) else None,
                    "end_value": float(x[end]) if np.isfinite(x[end]) else None,
                })

        print(f"[完成] feature_{feature:02d}")

    mask_path = out_dir / "local_outlier_mask.npy"
    summary_path = out_dir / "local_outlier_summary.csv"
    runs_path = out_dir / "local_outlier_runs.csv"

    np.save(mask_path, mask_all)

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(runs_path, "w", newline="", encoding="utf-8-sig") as f:
        if run_rows:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
        else:
            writer = csv.writer(f)
            writer.writerow(["feature", "batch", "start", "end", "length", "start_value", "end_value"])

    print("\n[完成]")
    print(f"异常掩码: {mask_path}")
    print(f"异常统计: {summary_path}")
    print(f"连续段统计: {runs_path}")


if __name__ == "__main__":
    main()
