"""
diagnose_local_outliers_v2.py

用途：
    全量局部异常检测，但加入限制条件，避免常量列/低方差列被误判。
    只检测，不修改数据。

核心方法：
    rolling median + rolling MAD

新增限制：
    1. 方差为 0 的列不标记异常
    2. 近零方差列可跳过
    3. 加入最小绝对偏差阈值 min_abs_delta，避免 MAD 太小时过敏
    4. 可选择跳过低唯一值/离散状态列
    5. 输出 mask、summary、runs、feature_profile，方便后续清洗脚本复用

输入：
    train.npy 或 cleaned_train.npy
    支持 shape:
        (600000, 65)
        (65, 10, batch_len)

输出：
    reports/local_outliers_v2/
        local_outlier_mask.npy
        local_outlier_summary.csv
        local_outlier_runs.csv
        feature_profile.csv

运行：
    python .\\scripts\\diagnose_local_outliers_v2.py
"""

from pathlib import Path
import csv
import json

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


def ask_yes_no(prompt: str, default: bool) -> bool:
    default_text = "Y" if default else "N"
    value = input(f"{prompt}\n默认: {default_text}，输入 y/n，直接回车使用默认值: ").strip().lower()
    if not value:
        return default
    return value in ["y", "yes", "1", "true", "是"]


def to_batch_format(data: np.ndarray) -> np.ndarray:
    """
    return shape = (65, 10, batch_len)
    """
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


def get_feature_profile(data_3d: np.ndarray, near_zero_var_eps: float, low_unique_threshold: int) -> list[dict]:
    rows = []
    flat = data_3d.reshape(data_3d.shape[0], -1)

    for feature in range(data_3d.shape[0]):
        x = flat[feature]
        finite = x[np.isfinite(x)]

        if len(finite) == 0:
            var = np.nan
            std = np.nan
            n_unique = 0
        else:
            var = float(np.nanvar(finite))
            std = float(np.nanstd(finite))
            n_unique = int(len(np.unique(finite)))

        is_zero_var = bool(np.isfinite(var) and var == 0)
        is_near_zero_var = bool(np.isfinite(var) and var <= near_zero_var_eps)
        is_low_unique = bool(n_unique <= low_unique_threshold)

        rows.append({
            "feature": feature,
            "variance": var,
            "std": std,
            "n_unique": n_unique,
            "is_zero_var": is_zero_var,
            "is_near_zero_var": is_near_zero_var,
            "is_low_unique": is_low_unique,
        })

    return rows


def rolling_mad_outlier_mask(
    x: np.ndarray,
    window: int,
    threshold: float,
    min_abs_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    局部异常检测：
        med = rolling median
        mad = rolling median(abs(x - med))
        abs_dev = abs(x - med)
        score = abs_dev / (1.4826 * mad)

    判定条件：
        score > threshold
        且 abs_dev > min_abs_delta

    返回：
        mask, rolling_median, abs_dev
    """
    s = pd.Series(x.astype(float))

    min_periods = max(5, window // 5)

    med = s.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).median()

    abs_dev = (s - med).abs()

    mad = abs_dev.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).median()

    global_scale = np.nanmedian(np.abs(x - np.nanmedian(x)))
    fallback = global_scale if np.isfinite(global_scale) and global_scale > 1e-8 else 1e-6

    mad = mad.fillna(fallback)
    mad = mad.mask(mad < 1e-8, fallback)

    score = abs_dev / (1.4826 * mad)

    mask_score = score.gt(threshold).fillna(False).to_numpy(dtype=bool).copy()
    mask_abs = abs_dev.gt(min_abs_delta).fillna(False).to_numpy(dtype=bool).copy()
    mask_bad_value = ~np.isfinite(x)

    mask = (mask_score & mask_abs) | mask_bad_value

    return (
        mask.astype(bool),
        med.to_numpy(dtype=float),
        abs_dev.to_numpy(dtype=float),
    )


def find_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """
    返回 [(start, end, length)]，end 为闭区间
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    runs = []
    start = int(idx[0])
    prev = int(idx[0])

    for current_raw in idx[1:]:
        current = int(current_raw)
        if current == prev + 1:
            prev = current
        else:
            runs.append((start, prev, prev - start + 1))
            start = current
            prev = current

    runs.append((start, prev, prev - start + 1))
    return runs


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    print("=" * 72)
    print("局部异常诊断 v2：全量检测 + 方差/低方差/最小偏差限制")
    print("只检测，不清洗")
    print("=" * 72)

    input_path = ask_path("请输入要诊断的数据路径", "data/raw/train.npy")
    out_dir = ask_path("请输入输出目录", "reports/local_outliers_v2")

    window = ask_int("rolling 窗口大小，建议 101 或 201", 201)
    threshold = ask_float("MAD 异常阈值，建议 6~8，越大越保守", 8.0)

    near_zero_var_eps = ask_float("近零方差阈值；方差 <= 该值时不标记异常", 1e-10)
    min_abs_std_ratio = ask_float("最小绝对偏差阈值 = 每列 std * 该比例，建议 0.005~0.02", 0.01)
    min_abs_floor = ask_float("最小绝对偏差阈值下限，避免极小数值误判", 1e-6)

    skip_low_unique = ask_yes_no("是否跳过低唯一值/离散状态列", False)
    low_unique_threshold = ask_int("低唯一值阈值；n_unique <= 该值视为低唯一值列", 10)

    out_dir.mkdir(parents=True, exist_ok=True)

    data = to_batch_format(np.load(input_path))
    n_features, n_batches, batch_len = data.shape
    print(f"[读取] data shape = {data.shape}")

    profile_rows = get_feature_profile(
        data,
        near_zero_var_eps=near_zero_var_eps,
        low_unique_threshold=low_unique_threshold,
    )

    profile_by_feature = {row["feature"]: row for row in profile_rows}

    mask_all = np.zeros_like(data, dtype=bool)
    summary_rows = []
    run_rows = []

    for feature in range(n_features):
        profile = profile_by_feature[feature]
        feature_std = profile["std"]

        skip_reason = ""
        should_skip = False

        if profile["is_zero_var"]:
            should_skip = True
            skip_reason = "zero_variance"
        elif profile["is_near_zero_var"]:
            should_skip = True
            skip_reason = "near_zero_variance"
        elif skip_low_unique and profile["is_low_unique"]:
            should_skip = True
            skip_reason = "low_unique"

        if np.isfinite(feature_std):
            min_abs_delta = max(float(feature_std) * min_abs_std_ratio, min_abs_floor)
        else:
            min_abs_delta = min_abs_floor

        for batch in range(n_batches):
            x = data[feature, batch, :]

            if should_skip:
                mask = np.zeros(batch_len, dtype=bool)
                runs = []
            else:
                mask, _, _ = rolling_mad_outlier_mask(
                    x,
                    window=window,
                    threshold=threshold,
                    min_abs_delta=min_abs_delta,
                )
                runs = find_runs(mask)
                mask_all[feature, batch, :] = mask

            lengths = [r[2] for r in runs]

            summary_rows.append({
                "feature": feature,
                "batch": batch,
                "skipped": bool(should_skip),
                "skip_reason": skip_reason,
                "variance": profile["variance"],
                "std": profile["std"],
                "n_unique": profile["n_unique"],
                "min_abs_delta": min_abs_delta,
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

        print(
            f"[完成] feature_{feature:02d} "
            f"{'(skip: ' + skip_reason + ')' if should_skip else ''}"
        )

    mask_path = out_dir / "local_outlier_mask.npy"
    summary_path = out_dir / "local_outlier_summary.csv"
    runs_path = out_dir / "local_outlier_runs.csv"
    profile_path = out_dir / "feature_profile.csv"
    config_path = out_dir / "diagnose_config.json"

    np.save(mask_path, mask_all)
    write_csv(summary_path, summary_rows)
    write_csv(runs_path, run_rows)
    write_csv(profile_path, profile_rows)

    config = {
        "script": "diagnose_local_outliers_v2.py",
        "input_path": str(input_path),
        "data_shape": list(data.shape),
        "window": window,
        "threshold": threshold,
        "near_zero_var_eps": near_zero_var_eps,
        "min_abs_std_ratio": min_abs_std_ratio,
        "min_abs_floor": min_abs_floor,
        "skip_low_unique": skip_low_unique,
        "low_unique_threshold": low_unique_threshold,
        "outputs": {
            "mask": str(mask_path),
            "summary": str(summary_path),
            "runs": str(runs_path),
            "feature_profile": str(profile_path),
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    total_outliers = int(mask_all.sum())
    total_points = int(mask_all.size)
    print("\n[完成]")
    print(f"异常掩码: {mask_path}")
    print(f"异常统计: {summary_path}")
    print(f"连续段统计: {runs_path}")
    print(f"特征画像: {profile_path}")
    print(f"配置记录: {config_path}")
    print(f"总异常点: {total_outliers} / {total_points} = {total_outliers / total_points:.6%}")


if __name__ == "__main__":
    main()
