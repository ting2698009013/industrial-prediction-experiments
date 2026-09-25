"""
"中控杯"工业大模型初赛训练集 数据分析脚本
==========================================
功能：对 .npy 数据集进行全面探索性数据分析 (EDA)
输出：分类保存到 output/ 目录下的各个子文件夹，最终生成 report.md 汇总报告
"""

import numpy as np
import pandas as pd
import os
import glob
import sys
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 无需GUI后端
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 输出子目录
DIRS = {
    "basic_info": os.path.join(OUTPUT_DIR, "01_basic_info"),
    "missing_values": os.path.join(OUTPUT_DIR, "02_missing_values"),
    "distributions": os.path.join(OUTPUT_DIR, "03_distributions"),
    "correlation": os.path.join(OUTPUT_DIR, "04_correlation"),
    "outliers": os.path.join(OUTPUT_DIR, "05_outliers"),
    "variance": os.path.join(OUTPUT_DIR, "06_variance"),
    "sample_data": os.path.join(OUTPUT_DIR, "07_sample_data"),
}

# Matplotlib 中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 特征名称（65列，暂时用 feature_0 ~ feature_64）
N_FEATURES = 65
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(N_FEATURES)]


def load_data():
    """加载数据集"""
    npy_files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    if not npy_files:
        print("错误：未找到 .npy 文件！")
        sys.exit(1)
    filepath = npy_files[0]
    print(f"[加载] 文件: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    print(f"[加载] 形状: {data.shape}, 类型: {data.dtype}")
    return data, filepath


def create_dirs():
    """创建所有输出目录"""
    for name, path in DIRS.items():
        os.makedirs(path, exist_ok=True)
    print(f"[目录] 输出目录已创建: {OUTPUT_DIR}")


def analyze_basic_info(data, filepath):
    """1. 基本信息分析"""
    print("\n" + "=" * 60)
    print("[分析1] 基本信息")
    print("=" * 60)

    info = {
        "文件路径": filepath,
        "数组形状": str(data.shape),
        "数据类型": str(data.dtype),
        "样本数": data.shape[0],
        "特征数": data.shape[1],
        "总元素数": data.size,
        "内存占用(MB)": round(data.nbytes / (1024 * 1024), 2),
    }

    for k, v in info.items():
        print(f"  {k}: {v}")

    # 每列基本统计（使用 nan-safe 函数）
    nan_counts_per_col = np.isnan(data).sum(axis=0)
    valid_counts = data.shape[0] - nan_counts_per_col

    stats_dict = {
        "特征名": FEATURE_NAMES,
        "数据类型": [str(data.dtype)] * N_FEATURES,
        "非NaN数量": valid_counts,
        "NaN数量": nan_counts_per_col,
        "NaN比例(%)": np.round(nan_counts_per_col / data.shape[0] * 100, 4),
        "最小值": np.nanmin(data, axis=0),
        "最大值": np.nanmax(data, axis=0),
        "均值": np.nanmean(data, axis=0),
        "标准差": np.nanstd(data, axis=0),
        "中位数": np.nanmedian(data, axis=0),
        "Q1(25%)": np.nanpercentile(data, 25, axis=0),
        "Q3(75%)": np.nanpercentile(data, 75, axis=0),
    }
    # 极差单独计算（nanmin/nanmax 返回的是标量数组）
    stats_dict["极差"] = np.nanmax(data, axis=0) - np.nanmin(data, axis=0)

    df_stats = pd.DataFrame(stats_dict)
    # 四舍五入
    for col in [
        "最小值",
        "最大值",
        "均值",
        "标准差",
        "中位数",
        "Q1(25%)",
        "Q3(75%)",
        "极差",
    ]:
        df_stats[col] = df_stats[col].round(4)

    save_path = os.path.join(DIRS["basic_info"], "basic_stats.csv")
    df_stats.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    return df_stats, info


def analyze_missing_values(data):
    """2. 缺失值分析"""
    print("\n" + "=" * 60)
    print("[分析2] 缺失值分析")
    print("=" * 60)

    nan_count = np.isnan(data).sum(axis=0)
    inf_count = np.isinf(data).sum(axis=0)
    total = data.shape[0]

    missing_dict = {
        "特征名": FEATURE_NAMES,
        "NaN数量": nan_count,
        "NaN比例(%)": (nan_count / total * 100).round(4),
        "Inf数量": inf_count,
        "Inf比例(%)": (inf_count / total * 100).round(4),
    }

    df_missing = pd.DataFrame(missing_dict)

    save_path = os.path.join(DIRS["missing_values"], "missing_report.csv")
    df_missing.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    total_nan = nan_count.sum()
    total_inf = inf_count.sum()
    print(f"  总NaN数量: {total_nan}, 总Inf数量: {total_inf}")

    # 可视化
    if total_nan > 0 or total_inf > 0:
        fig, ax = plt.subplots(figsize=(16, 6))
        x = np.arange(N_FEATURES)
        width = 0.35
        ax.bar(x - width / 2, nan_count, width, label="NaN", color="red", alpha=0.7)
        ax.bar(x + width / 2, inf_count, width, label="Inf", color="orange", alpha=0.7)
        ax.set_xlabel("特征")
        ax.set_ylabel("数量")
        ax.set_title("各特征缺失值/异常值数量")
        ax.set_xticks(x)
        ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
        ax.legend()
        ax.set_yscale("symlog")
        plt.tight_layout()
        fig_path = os.path.join(DIRS["missing_values"], "missing_visualization.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  [保存] {fig_path}")
    else:
        print("  数据集中无 NaN 或 Inf 值，跳过可视化。")

    return df_missing


def analyze_distributions(data):
    """3. 特征分布分析"""
    print("\n" + "=" * 60)
    print("[分析3] 特征分布分析")
    print("=" * 60)

    save_dir = DIRS["distributions"]

    for i in range(N_FEATURES):
        col_data = data[:, i]
        # 过滤 NaN 用于绘图
        valid_data = col_data[~np.isnan(col_data)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 直方图
        axes[0].hist(
            valid_data, bins=80, color="steelblue", edgecolor="white", alpha=0.8
        )
        axes[0].set_title(f"{FEATURE_NAMES[i]} 分布直方图")
        axes[0].set_xlabel("值")
        axes[0].set_ylabel("频次")
        col_mean = np.nanmean(col_data)
        col_median = np.nanmedian(col_data)
        axes[0].axvline(
            col_mean, color="red", linestyle="--", label=f"均值={col_mean:.2f}"
        )
        axes[0].axvline(
            col_median, color="green", linestyle="--", label=f"中位数={col_median:.2f}"
        )
        axes[0].legend(fontsize=8)

        # 箱线图（用有效数据）
        bp = axes[1].boxplot(
            valid_data,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="navy"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="navy"),
            capprops=dict(color="navy"),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=2, alpha=0.3),
        )
        axes[1].set_title(f"{FEATURE_NAMES[i]} 箱线图")
        axes[1].set_ylabel("值")

        plt.tight_layout()
        fig_path = os.path.join(save_dir, f"{FEATURE_NAMES[i]}_dist.png")
        plt.savefig(fig_path, dpi=100)
        plt.close()

        if (i + 1) % 10 == 0 or i == N_FEATURES - 1:
            print(f"  已完成 {i+1}/{N_FEATURES} 个特征分布图")

    print(f"  [保存] 分布图保存至: {save_dir}")

    # 偏度和峰度统计（逐列计算，使用 nan_policy='omit' 跳过NaN）
    from scipy import stats as sp_stats

    skewness = np.array(
        [sp_stats.skew(data[:, i], nan_policy="omit") for i in range(N_FEATURES)]
    )
    kurtosis_vals = np.array(
        [sp_stats.kurtosis(data[:, i], nan_policy="omit") for i in range(N_FEATURES)]
    )

    dist_stats = pd.DataFrame(
        {
            "特征名": FEATURE_NAMES,
            "偏度": np.round(skewness, 4),
            "峰度": np.round(kurtosis_vals, 4),
        }
    )

    dist_path = os.path.join(save_dir, "distribution_stats.csv")
    dist_stats.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {dist_path}")

    return dist_stats


def analyze_correlation(data):
    """4. 相关性分析"""
    print("\n" + "=" * 60)
    print("[分析4] 相关性分析")
    print("=" * 60)

    # 计算相关系数矩阵（用 pandas 方便计算）
    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    corr_matrix = df.corr()

    # 保存 CSV
    save_path = os.path.join(DIRS["correlation"], "correlation_matrix.csv")
    corr_matrix.to_csv(save_path, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    # 热力图
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(N_FEATURES))
    ax.set_yticks(range(N_FEATURES))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
    ax.set_title("特征相关系数热力图", fontsize=16)
    plt.colorbar(im, ax=ax, shrink=0.8, label="相关系数")
    plt.tight_layout()
    fig_path = os.path.join(DIRS["correlation"], "correlation_heatmap.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [保存] {fig_path}")

    # 找出高相关性特征对
    high_corr_pairs = []
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.8:
                high_corr_pairs.append(
                    {
                        "特征A": FEATURE_NAMES[i],
                        "特征B": FEATURE_NAMES[j],
                        "相关系数": round(val, 4),
                    }
                )

    if high_corr_pairs:
        df_high_corr = pd.DataFrame(high_corr_pairs)
        df_high_corr = df_high_corr.sort_values(by="相关系数", key=abs, ascending=False)
        hc_path = os.path.join(DIRS["correlation"], "high_correlation_pairs.csv")
        df_high_corr.to_csv(hc_path, index=False, encoding="utf-8-sig")
        print(f"  发现 {len(high_corr_pairs)} 对高相关性特征对 (|r| > 0.8)")
        print(f"  [保存] {hc_path}")
    else:
        df_high_corr = pd.DataFrame()
        print("  未发现 |相关系数| > 0.8 的特征对")

    return corr_matrix, df_high_corr


def analyze_outliers(data):
    """5. 异常值检测 (IQR 方法，排除NaN)"""
    print("\n" + "=" * 60)
    print("[分析5] 异常值检测")
    print("=" * 60)

    # 使用 nanpercentile 排除 NaN
    Q1 = np.nanpercentile(data, 25, axis=0)
    Q3 = np.nanpercentile(data, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_counts = []
    nan_counts = []
    for i in range(N_FEATURES):
        col_data = data[:, i]
        valid_mask = ~np.isnan(col_data)
        nan_count = (~valid_mask).sum()
        nan_counts.append(nan_count)
        # 只在非NaN数据中计算异常值
        outliers = (
            (col_data < lower_bound[i]) | (col_data > upper_bound[i])
        ) & valid_mask
        outlier_counts.append(outliers.sum())

    outlier_arr = np.array(outlier_counts)
    nan_arr = np.array(nan_counts)
    valid_total = data.shape[0] - nan_arr  # 每列有效数据量

    df_outliers = pd.DataFrame(
        {
            "特征名": FEATURE_NAMES,
            "Q1": np.round(Q1, 4),
            "Q3": np.round(Q3, 4),
            "IQR": np.round(IQR, 4),
            "下界(Q1-1.5*IQR)": np.round(lower_bound, 4),
            "上界(Q3+1.5*IQR)": np.round(upper_bound, 4),
            "异常值数量": outlier_arr,
            "异常值比例(%)": np.round(outlier_arr / valid_total * 100, 4),
        }
    )

    save_path = os.path.join(DIRS["outliers"], "outlier_report.csv")
    df_outliers.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    # 部分特征箱线图（选取异常值最多的前16个特征，排除NaN）
    top_outlier_idx = np.argsort(outlier_arr)[::-1][:16]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    for k, idx in enumerate(top_outlier_idx):
        if outlier_arr[idx] == 0:
            break
        valid_col = data[:, idx][~np.isnan(data[:, idx])]
        axes[k].boxplot(
            valid_col,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightyellow", color="navy"),
            medianprops=dict(color="red", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="red", markersize=1, alpha=0.2),
        )
        axes[k].set_title(f"{FEATURE_NAMES[idx]} (异常:{outlier_arr[idx]})", fontsize=9)
        axes[k].tick_params(labelsize=7)

    plt.suptitle("异常值最多的16个特征箱线图", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(DIRS["outliers"], "boxplot_top_outliers.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [保存] {fig_path}")

    return df_outliers


def analyze_variance(data):
    """6. 方差分析（排除NaN）"""
    print("\n" + "=" * 60)
    print("[分析6] 方差分析")
    print("=" * 60)

    # 使用 nan-safe 函数
    variances = np.nanvar(data, axis=0)
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)

    # 变异系数 (CV) — 避免除以零
    abs_means = np.abs(means)
    cv = np.where(abs_means > 1e-8, stds / abs_means, 0.0)

    df_var = pd.DataFrame(
        {
            "特征名": FEATURE_NAMES,
            "方差": np.round(variances.astype(float), 6),
            "标准差": np.round(stds.astype(float), 4),
            "变异系数(CV)": np.round(cv.astype(float), 4),
            "均值": np.round(means.astype(float), 4),
        }
    )

    df_var = df_var.sort_values(by="方差", ascending=True).reset_index(drop=True)

    save_path = os.path.join(DIRS["variance"], "variance_report.csv")
    df_var.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    # 方差排序可视化（过滤掉 NaN 方差的特征）
    valid_var = df_var.dropna(subset=["方差"])
    var_values = valid_var["方差"].values.astype(float)

    fig, ax = plt.subplots(figsize=(16, 6))
    # 识别零方差特征
    colors = [
        (
            "red"
            if v == 0
            else ("orange" if v < np.nanpercentile(var_values, 10) else "green")
        )
        for v in var_values
    ]
    ax.barh(range(len(valid_var)), var_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(valid_var)))
    ax.set_yticklabels(valid_var["特征名"].values, fontsize=7)
    ax.set_xlabel("方差")
    ax.set_title("各特征方差排序（红色=零方差，橙色=低方差）")
    plt.tight_layout()
    fig_path = os.path.join(DIRS["variance"], "variance_barplot.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [保存] {fig_path}")

    # 识别低方差和零方差特征
    zero_var_features = df_var[df_var["方差"] == 0]
    low_var_threshold = np.nanpercentile(var_values, 10)
    low_var_features = df_var[
        (df_var["方差"] > 0) & (df_var["方差"] < low_var_threshold)
    ]
    print(f"  零方差特征数量: {len(zero_var_features)}")
    print(f"  低方差阈值 (P10): {low_var_threshold:.6f}")
    print(f"  低方差特征数量: {len(low_var_features)}")

    return df_var, zero_var_features, low_var_features


def save_sample_data(data):
    """7. 保存部分样本数据"""
    print("\n" + "=" * 60)
    print("[分析7] 样本数据")
    print("=" * 60)

    df_sample = pd.DataFrame(data[:100], columns=FEATURE_NAMES)
    save_path = os.path.join(DIRS["sample_data"], "first_100_samples.csv")
    df_sample.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [保存] {save_path}")

    return df_sample


def generate_report(
    info,
    df_stats,
    df_missing,
    dist_stats,
    corr_matrix,
    df_high_corr,
    df_outliers,
    df_var,
    zero_var_features,
    low_var_features,
):
    """生成最终汇总报告 (Markdown)"""
    print("\n" + "=" * 60)
    print("[报告] 生成汇总报告")
    print("=" * 60)

    lines = []
    lines.append(f'# "中控杯"工业大模型初赛训练集 数据分析报告\n')
    lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # === 基本信息 ===
    lines.append("---\n## 1. 数据集基本信息\n")
    lines.append(f"| 属性 | 值 |")
    lines.append(f"|------|------|")
    for k, v in info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # === 缺失值 ===
    lines.append("---\n## 2. 缺失值分析\n")
    total_nan = df_missing["NaN数量"].sum()
    total_inf = df_missing["Inf数量"].sum()
    lines.append(f"- **总NaN数量**: {total_nan}")
    lines.append(f"- **总Inf数量**: {total_inf}")
    if total_nan == 0 and total_inf == 0:
        lines.append("\n✅ 数据集无缺失值和无穷值，数据质量良好。\n")
    else:
        lines.append(f"\n⚠️ 数据存在缺失或异常值，需进一步处理。\n")
        has_nan = df_missing[df_missing["NaN数量"] > 0]
        if len(has_nan) > 0:
            lines.append("**含NaN的特征**:\n")
            lines.append(has_nan.to_markdown(index=False))
            lines.append("")

    # === 基本统计 ===
    lines.append("---\n## 3. 各特征基本统计\n")
    lines.append("> 完整统计见 `output/01_basic_info/basic_stats.csv`\n")
    lines.append(
        df_stats[
            ["特征名", "最小值", "最大值", "均值", "标准差", "中位数"]
        ].to_markdown(index=False)
    )
    lines.append("")

    # === 特征分布 ===
    lines.append("---\n## 4. 特征分布分析\n")
    lines.append("> 各特征直方图和箱线图见 `output/03_distributions/` 文件夹\n")
    lines.append("**偏度和峰度统计**:\n")
    lines.append("> 完整统计见 `output/03_distributions/distribution_stats.csv`\n")

    # 偏度绝对值较大的特征
    if dist_stats is not None:
        high_skew = dist_stats[abs(dist_stats["偏度"]) > 1]
        if len(high_skew) > 0:
            lines.append(f"**偏度绝对值 > 1 的特征** ({len(high_skew)}个):\n")
            lines.append(high_skew.to_markdown(index=False))
            lines.append("")

    # === 相关性 ===
    lines.append("---\n## 5. 相关性分析\n")
    lines.append("> 相关系数矩阵见 `output/04_correlation/correlation_matrix.csv`\n")
    lines.append("> 热力图见 `output/04_correlation/correlation_heatmap.png`\n")
    if len(df_high_corr) > 0:
        lines.append(f"**高相关性特征对 (|r| > 0.8)**: 共 {len(df_high_corr)} 对\n")
        lines.append("> 详情见 `output/04_correlation/high_correlation_pairs.csv`\n")
        # 显示前20对
        top_pairs = df_high_corr.head(20)
        lines.append(top_pairs.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("未发现 |相关系数| > 0.8 的高相关特征对。\n")

    # === 异常值 ===
    lines.append("---\n## 6. 异常值分析\n")
    lines.append("> 完整报告见 `output/05_outliers/outlier_report.csv`\n")
    has_outliers = df_outliers[df_outliers["异常值数量"] > 0]
    lines.append(f"- **含异常值的特征数量**: {len(has_outliers)} / {N_FEATURES}")
    total_outliers = has_outliers["异常值数量"].sum()
    lines.append(f"- **异常值总数量**: {total_outliers}")
    lines.append(
        f'- **异常值占总数据比例**: {total_outliers / (info["样本数"] * N_FEATURES) * 100:.4f}%\n'
    )

    if len(has_outliers) > 0:
        top_outlier_features = has_outliers.sort_values(
            "异常值数量", ascending=False
        ).head(15)
        lines.append("**异常值最多的15个特征**:\n")
        lines.append(
            top_outlier_features[
                [
                    "特征名",
                    "异常值数量",
                    "异常值比例(%)",
                    "下界(Q1-1.5*IQR)",
                    "上界(Q3+1.5*IQR)",
                ]
            ].to_markdown(index=False)
        )
        lines.append("")

    # === 方差分析 ===
    lines.append("---\n## 7. 方差分析\n")
    lines.append("> 完整报告见 `output/06_variance/variance_report.csv`\n")

    if len(zero_var_features) > 0:
        lines.append(f"**零方差（常量）特征** ({len(zero_var_features)}个):\n")
        lines.append(
            zero_var_features[["特征名", "方差", "均值"]].to_markdown(
                index=False
            )
        )
        lines.append("\n⚠️ 这些特征为常量值，对模型无任何贡献，建议直接剔除。\n")

    if len(low_var_features) > 0:
        lines.append(f"**低方差特征** ({len(low_var_features)}个):\n")
        lines.append(
            low_var_features[["特征名", "方差", "变异系数(CV)"]].to_markdown(
                index=False
            )
        )
        lines.append("\n⚠️ 低方差特征可能对模型贡献较小，可考虑特征选择时优先剔除。\n")

    # === 数据概览与建议 ===
    lines.append("---\n## 8. 数据概览与建议\n")
    lines.append(
        f'1. **数据规模**: 共 {info["样本数"]:,} 条样本, {info["特征数"]} 个特征'
    )
    lines.append(
        f'2. **数据质量**: {"✅ 无缺失值" if total_nan == 0 else "⚠️ 存在缺失值"}'
    )

    if len(df_high_corr) > 0:
        lines.append(
            f"3. **特征冗余**: 存在 {len(df_high_corr)} 对高相关特征对，建议考虑降维或特征选择"
        )

    if total_outliers > 0:
        lines.append(
            f'4. **异常值**: 共 {total_outliers} 个异常值，占比 {total_outliers / (info["样本数"] * N_FEATURES) * 100:.4f}%，需关注对模型的影响'
        )

    lines.append(f"5. **建议后续步骤**: 特征工程、特征选择、模型选择与训练")
    lines.append("")

    # === 文件结构 ===
    lines.append("---\n## 9. 输出文件结构\n")
    lines.append("```\n")
    lines.append("output/\n")
    lines.append("├── 01_basic_info/\n")
    lines.append("│   └── basic_stats.csv          # 各特征基本统计\n")
    lines.append("├── 02_missing_values/\n")
    lines.append("│   └── missing_report.csv       # 缺失值报告\n")
    lines.append("├── 03_distributions/\n")
    lines.append("│   ├── feature_XX_dist.png      # 各特征分布图\n")
    lines.append("│   └── distribution_stats.csv   # 偏度峰度统计\n")
    lines.append("├── 04_correlation/\n")
    lines.append("│   ├── correlation_matrix.csv   # 相关系数矩阵\n")
    lines.append("│   ├── correlation_heatmap.png  # 热力图\n")
    lines.append("│   └── high_correlation_pairs.csv\n")
    lines.append("├── 05_outliers/\n")
    lines.append("│   ├── outlier_report.csv       # 异常值报告\n")
    lines.append("│   └── boxplot_top_outliers.png # 异常值最多特征箱线图\n")
    lines.append("├── 06_variance/\n")
    lines.append("│   ├── variance_report.csv      # 方差报告\n")
    lines.append("│   └── variance_barplot.png     # 方差排序图\n")
    lines.append("├── 07_sample_data/\n")
    lines.append("│   └── first_100_samples.csv    # 前100条样本\n")
    lines.append("└── report.md                    # 本报告\n")
    lines.append("```\n")

    report_text = "\n".join(lines)

    save_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  [保存] {save_path}")
    return report_text


def main():
    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print('║    "中控杯"工业大模型初赛训练集 数据分析工具           ║')
    print("╚══════════════════════════════════════════════════════════╝")

    # 创建输出目录
    create_dirs()

    # 加载数据
    data, filepath = load_data()

    # 1. 基本信息
    df_stats, info = analyze_basic_info(data, filepath)

    # 2. 缺失值
    df_missing = analyze_missing_values(data)

    # 3. 分布
    dist_stats = analyze_distributions(data)

    # 4. 相关性
    corr_matrix, df_high_corr = analyze_correlation(data)

    # 5. 异常值
    df_outliers = analyze_outliers(data)

    # 6. 方差
    df_var, zero_var_features, low_var_features = analyze_variance(data)

    # 7. 样本数据
    save_sample_data(data)

    # 生成汇总报告
    generate_report(
        info,
        df_stats,
        df_missing,
        dist_stats,
        corr_matrix,
        df_high_corr,
        df_outliers,
        df_var,
        zero_var_features,
        low_var_features,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 分析完成！总耗时: {elapsed:.1f} 秒")
    print(f"📁 结果保存在: {OUTPUT_DIR}")
    print(f"📊 汇总报告: {os.path.join(OUTPUT_DIR, 'report.md')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
