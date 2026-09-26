#!/usr/bin/env python3
"""Reproducible miniature benchmark for the industrial forecasting workflow.

The original competition data cannot be distributed. This script creates a
small synthetic process in which future exogenous sensor values are known and
the final target window is hidden, matching the shape of the original task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_NAMES = ("feature_62", "feature_63", "feature_64")


def make_process(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sensors = np.zeros((rows, 12), dtype=np.float64)
    innovations = rng.normal(0.0, 1.0, size=sensors.shape)
    for index in range(1, rows):
        sensors[index] = 0.82 * sensors[index - 1] + innovations[index]

    time = np.arange(rows, dtype=np.float64)
    mode = ((time // 180) % 3).astype(int)
    mode_features = np.eye(3, dtype=np.float64)[mode]
    periodic = np.column_stack((
        np.sin(time / 45.0),
        np.cos(time / 45.0),
        np.sin(time / 130.0),
        np.cos(time / 130.0),
    ))
    features = np.column_stack((sensors, mode_features, periodic))

    noise = rng.normal(0.0, 0.12, size=(rows, 3))
    targets = np.column_stack((
        1.7 * sensors[:, 0] - 0.8 * sensors[:, 3] + 0.5 * sensors[:, 7]
        + 1.2 * mode_features[:, 1] + periodic[:, 0],
        -1.1 * sensors[:, 1] + 0.7 * sensors[:, 5] + 0.3 * sensors[:, 8]
        - 0.9 * mode_features[:, 2] + 0.6 * periodic[:, 2],
        0.6 * sensors[:, 2] * sensors[:, 4] + 0.9 * sensors[:, 9]
        + 0.8 * mode_features[:, 0] - 0.4 * periodic[:, 1],
    )) + noise
    return features, targets


def metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(expected, predicted)),
        "rmse": float(mean_squared_error(expected, predicted) ** 0.5),
    }


def run_benchmark(rows: int, horizon: int, seed: int) -> dict[str, object]:
    features, targets = make_process(rows, seed)
    split = rows - horizon
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]

    predictions: dict[str, np.ndarray] = {
        "persistence": np.repeat(train_y[-1][None, :], horizon, axis=0),
    }
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(train_x, train_y)
    predictions["ridge"] = ridge.predict(test_x)

    boosted_columns = []
    for target_index in range(train_y.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=0.1,
            random_state=seed,
        )
        model.fit(train_x, train_y[:, target_index])
        boosted_columns.append(model.predict(test_x))
    predictions["hist_gradient_boosting"] = np.column_stack(boosted_columns)

    model_results: dict[str, object] = {}
    for model_name, prediction in predictions.items():
        per_target = {
            name: metrics(test_y[:, index], prediction[:, index])
            for index, name in enumerate(TARGET_NAMES)
        }
        model_results[model_name] = {
            "overall": metrics(test_y, prediction),
            "per_target": per_target,
        }

    if model_results["ridge"]["overall"]["mae"] >= model_results["persistence"]["overall"]["mae"]:
        raise RuntimeError("ridge baseline unexpectedly failed to beat persistence")

    return {
        "dataset": {"rows": rows, "training_rows": split, "forecast_horizon": horizon, "seed": seed},
        "models": model_results,
    }


def print_summary(results: dict[str, object]) -> None:
    print("Synthetic industrial forecasting benchmark")
    print("model                         MAE       RMSE")
    print("-" * 46)
    for name, result in results["models"].items():
        overall = result["overall"]
        print(f"{name:28s} {overall['mae']:8.4f} {overall['rmse']:10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=12_000)
    parser.add_argument("--horizon", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="use a smaller CI-friendly dataset")
    parser.add_argument("--output", type=Path, help="optional JSON metrics path")
    args = parser.parse_args()
    if args.quick:
        args.rows, args.horizon = 3_000, 500
    if args.rows <= args.horizon or args.horizon < 1:
        parser.error("rows must be greater than horizon, and horizon must be positive")

    results = run_benchmark(args.rows, args.horizon, args.seed)
    print_summary(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
