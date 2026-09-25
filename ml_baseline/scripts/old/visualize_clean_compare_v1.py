"""
visualize_clean_compare_v1.py

用途:
    对比原始训练集/清洗前数据 与 清洗后数据。
    每一列画图：清洗前红色，清洗后蓝色。
    支持按 batch 画图，默认每个特征每个 batch 生成一张图。

输入:
    raw train.npy:
        可以是 (600000, 65) 或 (65, 10, batch_len)
    cleaned_train.npy:
        应为 (65, 10, batch_len)，也支持 (600000, 65)

输出:
    reports/clean_compare_v1/
        feature_00_batch_00.png
        feature_00_batch_01.png
        ...
    reports/clean_compare_v1/compare_summary.csv

运行:
    python .\scripts\visualize_clean_compare_v1.py
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


N_BATCHES = 10
N_FEATURES = 65


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_int(prompt: str, default: int) -> int:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return int(value) if value else default


def ask_feature_list() -> list[int]:
    text = input(
        "请输入要画的特征列，例如 0,1,28,58；直接回车表示画全部 0~64:\n"
    ).strip()
    if not text:
        return list(range(N_FEATURES))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def to_batch_format(data: np.ndarray) -> np.ndarray:
    """return shape = (65, 10, batch_len)"""
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


def downsample_series(x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        idx = np.arange(n)
        return idx, x

    idx = np.linspace(0, n - 1, max_points).astype(int)
    return idx, x[idx]


def plot_one_feature_batch(raw_3d, clean_3d, feature, batch, out_dir, max_points):
    raw = raw_3d[feature, batch, :]
    clean = clean_3d[feature, batch, :]

    idx_raw, raw_plot = downsample_series(raw, max_points)
    idx_clean, clean_plot = downsample_series(clean, max_points)

    raw_finite = np.isfinite(raw)
    clean_finite = np.isfinite(clean)
    changed = raw_finite & clean_finite & (raw != clean)
    # 原始 NaN/Inf 被清洗成正常值，也算 changed
    changed |= (~raw_finite) & clean_finite
    n_changed = int(changed.sum())

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx_raw, raw_plot, color="red", linewidth=0.8, alpha=0.7, label="before/raw")
    ax.plot(idx_clean, clean_plot, color="blue", linewidth=0.8, alpha=0.7, label="after/cleaned")

    changed_idx = np.where(changed)[0]
    if len(changed_idx) > 0:
        if len(changed_idx) > 300:
            changed_idx = np.linspace(changed_idx[0], changed_idx[-1], 300).astype(int)
        ax.scatter(changed_idx, clean[changed_idx], s=8, color="black", alpha=0.6, label="changed points")

    ax.set_title(f"feature_{feature:02d} batch_{batch:02d} | changed={n_changed}")
    ax.set_xlabel("time index within batch")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.25)

    out_path = out_dir / f"feature_{feature:02d}_batch_{batch:02d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return {
        "feature": feature,
        "batch": batch,
        "raw_nan": int(np.isnan(raw).sum()),
        "clean_nan": int(np.isnan(clean).sum()),
        "raw_inf": int(np.isinf(raw).sum()),
        "clean_inf": int(np.isinf(clean).sum()),
        "changed_count": n_changed,
        "changed_ratio": n_changed / len(raw),
        "raw_mean": float(np.nanmean(raw)),
        "clean_mean": float(np.nanmean(clean)),
        "raw_std": float(np.nanstd(raw)),
        "clean_std": float(np.nanstd(clean)),
        "plot": str(out_path),
    }


def main():
    print("=" * 72)
    print("清洗前后可视化对比：红色=清洗前，蓝色=清洗后")
    print("=" * 72)

    raw_path = ask_path("请输入原始 train.npy 路径", "data/raw/train.npy")
    clean_path = ask_path("请输入清洗后 cleaned_train.npy 路径", "data/processed/cleaned_train_v1.npy")
    out_dir = ask_path("请输入图片输出目录", "reports/clean_compare_v1")
    max_points = ask_int("每张图最多绘制多少个点；太大图片会慢，建议 5000~10000", 8000)
    features = ask_feature_list()

    out_dir.mkdir(parents=True, exist_ok=True)

    raw = to_batch_format(np.load(raw_path))
    clean = to_batch_format(np.load(clean_path))

    if raw.shape != clean.shape:
        raise ValueError(f"原始数据和清洗数据形状不一致: raw={raw.shape}, clean={clean.shape}")

    print(f"[读取完成] raw shape={raw.shape}, clean shape={clean.shape}")

    rows = []
    for feature in features:
        for batch in range(raw.shape[1]):
            row = plot_one_feature_batch(raw, clean, feature, batch, out_dir, max_points)
            rows.append(row)
        print(f"[完成] feature_{feature:02d}")

    summary_path = out_dir / "compare_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[完成]")
    print(f"图片目录: {out_dir}")
    print(f"对比统计: {summary_path}")


if __name__ == "__main__":
    main()
