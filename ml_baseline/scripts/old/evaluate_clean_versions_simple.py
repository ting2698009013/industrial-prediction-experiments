"""
evaluate_clean_versions_simple.py

用途：
    用最简单的传统机器学习方法，比较不同清洗版本是否让预测效果变好。
    这不是正式建模，只是清洗效果验证。

默认验证方式：
    每个 batch 前 80% 训练，后 20% 验证
    X = 输入列 0~61，默认去掉严格常量列
    y = 目标列 62,63,64
    模型 = Ridge，可选 RandomForest 小样本验证

输入：
    data/processed/cleaned_train_v1.npy
    data/processed/cleaned_train_v2.npy
    也可以继续添加 v3 等版本

输出：
    reports/evaluate_clean_versions/evaluation_results.csv
    reports/evaluate_clean_versions/predictions_<version>.npy

运行：
    python .\\scripts\\evaluate_clean_versions_simple.py
"""

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


N_BATCHES = 10
N_FEATURES = 65
TARGET_COLS = [62, 63, 64]

# 只去掉严格常量列。35/60/61 先保留，让验证结果说话。
DEFAULT_DROP_INPUT_COLS = [5, 6, 39, 40, 41, 42, 43, 44, 45, 46]


def ask_path(prompt: str, default: str) -> Path:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return Path(value) if value else Path(default)


def ask_text(prompt: str, default: str) -> str:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return value if value else default


def ask_int(prompt: str, default: int) -> int:
    value = input(f"{prompt}\n默认: {default}\n直接回车使用默认值: ").strip()
    return int(value) if value else default


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


def parse_version_paths(text: str) -> list[tuple[str, Path]]:
    """
    输入格式：
        v1=data/processed/cleaned_train_v1.npy;v2=data/processed/cleaned_train_v2.npy
    """
    items = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"版本路径格式错误: {part}")
        name, path = part.split("=", 1)
        items.append((name.strip(), Path(path.strip())))
    return items


def build_xy(data_3d: np.ndarray, input_cols: list[int], cut_ratio: float):
    """
    data_3d: (65, 10, batch_len)

    输出：
        X_train, y_train, X_valid, y_valid
        y_valid_batch: (10, valid_len, 3)
    """
    batch_len = data_3d.shape[2]
    cut = int(batch_len * cut_ratio)

    # X: (batch, time, feature)
    X = data_3d[input_cols, :, :].transpose(1, 2, 0).astype(np.float32)
    y = data_3d[TARGET_COLS, :, :].transpose(1, 2, 0).astype(np.float32)

    X_train = X[:, :cut, :].reshape(-1, len(input_cols))
    y_train = y[:, :cut, :].reshape(-1, len(TARGET_COLS))

    X_valid = X[:, cut:, :].reshape(-1, len(input_cols))
    y_valid = y[:, cut:, :].reshape(-1, len(TARGET_COLS))

    return X_train, y_train, X_valid, y_valid, y[:, cut:, :], cut


def metric_rows(version: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []

    mse_all = mean_squared_error(y_true, y_pred)
    mae_all = mean_absolute_error(y_true, y_pred)

    rows.append({
        "version": version,
        "model": model_name,
        "target": "all",
        "mse": mse_all,
        "rmse": float(np.sqrt(mse_all)),
        "mae": mae_all,
    })

    for i, target_col in enumerate(TARGET_COLS):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rows.append({
            "version": version,
            "model": model_name,
            "target": f"feature_{target_col}",
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
        })

    return rows


def subsample_train(X: np.ndarray, y: np.ndarray, n: int, seed: int = 42):
    if n <= 0 or n >= len(X):
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def evaluate_one_version(
    version: str,
    path: Path,
    input_cols: list[int],
    cut_ratio: float,
    out_dir: Path,
    run_rf: bool,
    rf_train_sample: int,
):
    print(f"\n[读取] {version}: {path}")
    data = to_batch_format(np.load(path))
    print(f"shape = {data.shape}")

    X_train, y_train, X_valid, y_valid, y_valid_batch, cut = build_xy(
        data,
        input_cols=input_cols,
        cut_ratio=cut_ratio,
    )

    print(f"X_train={X_train.shape}, y_train={y_train.shape}, X_valid={X_valid.shape}, y_valid={y_valid.shape}")

    rows = []

    # Ridge：快速、稳定，用来做主比较
    print(f"[训练] {version} Ridge")
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_valid)

    pred_path = out_dir / f"predictions_{version}_ridge.npy"
    np.save(pred_path, pred.astype(np.float32))

    rows.extend(metric_rows(version, "ridge", y_valid, pred))

    # RandomForest：只作为补充，小样本训练，避免太慢
    if run_rf:
        print(f"[训练] {version} RandomForest，训练样本数上限={rf_train_sample}")
        X_rf, y_rf = subsample_train(X_train, y_train, rf_train_sample)

        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_rf, y_rf)
        pred_rf = rf.predict(X_valid)

        pred_rf_path = out_dir / f"predictions_{version}_random_forest.npy"
        np.save(pred_rf_path, pred_rf.astype(np.float32))

        rows.extend(metric_rows(version, "random_forest", y_valid, pred_rf))

    return rows, {
        "version": version,
        "path": str(path),
        "shape": list(data.shape),
        "cut_index": cut,
        "input_cols": input_cols,
        "n_input_cols": len(input_cols),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
    }


def main():
    print("=" * 72)
    print("清洗版本简单机器学习验证")
    print("=" * 72)

    default_versions = "v1=data/processed/cleaned_train_v1.npy;v2=data/processed/cleaned_train_v2.npy"
    version_text = ask_text(
        "请输入要比较的版本路径，格式 version=path;version=path",
        default_versions,
    )
    version_paths = parse_version_paths(version_text)

    out_dir = ask_path("请输入输出目录", "reports/evaluate_clean_versions")
    out_dir.mkdir(parents=True, exist_ok=True)

    cut_ratio_text = ask_text("训练比例，默认每个 batch 前 80% 训练", "0.8")
    cut_ratio = float(cut_ratio_text)

    drop_cols_text = ask_text(
        "输入列中要删除哪些列，逗号分隔；默认只删严格常量列",
        ",".join(map(str, DEFAULT_DROP_INPUT_COLS)),
    )
    drop_cols = set()
    if drop_cols_text.strip():
        drop_cols = {int(x.strip()) for x in drop_cols_text.split(",") if x.strip()}

    input_cols = [i for i in range(62) if i not in drop_cols]

    run_rf = ask_yes_no("是否额外运行 RandomForest 小样本验证；会比 Ridge 慢", False)
    rf_train_sample = ask_int("RandomForest 训练样本数上限", 80000)

    all_rows = []
    config_versions = []

    for version, path in version_paths:
        rows, info = evaluate_one_version(
            version=version,
            path=path,
            input_cols=input_cols,
            cut_ratio=cut_ratio,
            out_dir=out_dir,
            run_rf=run_rf,
            rf_train_sample=rf_train_sample,
        )
        all_rows.extend(rows)
        config_versions.append(info)

    result_path = out_dir / "evaluation_results.csv"
    with open(result_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    config = {
        "script": "evaluate_clean_versions_simple.py",
        "versions": config_versions,
        "cut_ratio": cut_ratio,
        "drop_cols": sorted(drop_cols),
        "input_cols": input_cols,
        "run_random_forest": run_rf,
        "rf_train_sample": rf_train_sample,
        "result_path": str(result_path),
    }

    config_path = out_dir / "evaluation_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"结果文件: {result_path}")
    print(f"配置文件: {config_path}")

    df = pd.DataFrame(all_rows)
    print("\n结果预览：")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
