"""
clean_train_v1.py

只负责训练集清洗，不做特征工程。

输入:
    data/raw/train.npy, shape = (600000, 65)

输出:
    data/processed/cleaned_train_v1.npy, shape = (65, 10, 60000)
    data/processed/clean_summary_v1.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

N_BATCHES = 10
N_FEATURES = 65


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_text(prompt: str, default: str) -> str:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return value if value else default


def load_and_to_batches(input_path: Path) -> np.ndarray:
    data = np.load(input_path)

    if data.ndim == 2:
        seq_len, n_features = data.shape
        if n_features != N_FEATURES:
            raise ValueError(f"二维训练集应有 {N_FEATURES} 个特征，但得到 {n_features}")

        batch_len = seq_len // N_BATCHES
        used_len = batch_len * N_BATCHES
        if used_len != seq_len:
            print(f"[提示] 原始长度 {seq_len} 不能被 {N_BATCHES} 整除，将截断到 {used_len}")

        data = data[:used_len]
        data = data.reshape(N_BATCHES, batch_len, n_features)
        data = data.transpose(2, 0, 1)  # (65, 10, batch_len)

    elif data.ndim == 3:
        if data.shape[0] != N_FEATURES:
            raise ValueError(f"三维数据第一维应为 {N_FEATURES} 个特征，但得到 {data.shape[0]}")
    else:
        raise ValueError(f"不支持的数据形状: {data.shape}")

    return data.astype(np.float32, copy=False)


def clean_batch(batch_2d: np.ndarray) -> tuple[np.ndarray, dict]:
    df = pd.DataFrame(batch_2d, columns=[f"feature_{i:02d}" for i in range(batch_2d.shape[1])])

    before_nan = int(df.isna().sum().sum())
    before_inf = int(np.isinf(df.to_numpy()).sum())

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.fillna(df.median(numeric_only=True)).fillna(0)

    after_nan = int(df.isna().sum().sum())
    after_inf = int(np.isinf(df.to_numpy()).sum())

    stats = {
        "nan_before": before_nan,
        "inf_before": before_inf,
        "nan_after": after_nan,
        "inf_after": after_inf,
    }

    return df.to_numpy(dtype=np.float32), stats


def clean_all_batches(data_3d: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    _, n_batches, _ = data_3d.shape
    cleaned = np.empty_like(data_3d, dtype=np.float32)
    batch_stats = []

    for b in range(n_batches):
        batch = data_3d[:, b, :].T
        cleaned_batch, stats = clean_batch(batch)
        cleaned[:, b, :] = cleaned_batch.T
        stats["batch"] = b
        batch_stats.append(stats)
        print(
            f"[清洗] batch {b}: NaN {stats['nan_before']} -> {stats['nan_after']}, "
            f"Inf {stats['inf_before']} -> {stats['inf_after']}"
        )

    return cleaned, batch_stats


def main():
    print("=" * 70)
    print("训练集清洗脚本 clean_train_v1.py")
    print("只做清洗，不做特征工程")
    print("=" * 70)

    input_path = ask_path("请输入原始 train.npy 路径", "data/raw/train.npy")
    out_dir = ask_path("请输入输出目录", "data/processed")
    prefix = ask_text("请输入版本前缀", "v1")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[读取] {input_path}")
    data_3d = load_and_to_batches(input_path)
    print(f"[形状] batch 格式: {data_3d.shape}")

    cleaned, batch_stats = clean_all_batches(data_3d)

    cleaned_path = out_dir / f"cleaned_train_{prefix}.npy"
    summary_path = out_dir / f"clean_summary_{prefix}.json"

    np.save(cleaned_path, cleaned)

    summary = {
        "script": "clean_train_v1.py",
        "version_prefix": prefix,
        "input_path": str(input_path),
        "cleaned_output_path": str(cleaned_path),
        "input_converted_shape": list(data_3d.shape),
        "cleaned_shape": list(cleaned.shape),
        "cleaning_steps": [
            "convert train.npy from (600000, 65) to (65, 10, 60000) if needed",
            "clean each batch independently",
            "replace inf and -inf with NaN",
            "ffill within each batch",
            "bfill within each batch",
            "fill remaining NaN with batch-column median",
            "fill remaining NaN with 0 as fallback",
        ],
        "note": "No columns are deleted in cleaned_train. Column deletion is handled in feature engineering.",
        "batch_stats": batch_stats,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"清洗数据: {cleaned_path}")
    print(f"清洗说明: {summary_path}")


if __name__ == "__main__":
    main()
