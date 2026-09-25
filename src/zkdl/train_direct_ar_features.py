from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SELECTED_COLS, TARGET_COLS, DL_TARGET_COLS, DL_TARGET_NAMES
from .dataset_ar_features import OfficialBatchWindowDatasetARFeatures
from .model import DualHeadTCNForecaster
from .scaler import Standardizer
from .utils import set_seed, save_json


def _as_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _make_visible_indices(n, n_batches, visible_ratio):
    if n % n_batches != 0:
        raise ValueError(f"len(data)={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))

    idx_parts = []
    for bi in range(n_batches):
        s = bi * batch_len
        idx_parts.append(np.arange(s, s + visible_len, dtype=np.int64))

    return np.concatenate(idx_parts), batch_len, visible_len


def fit_scalers(data, n_batches=10, visible_ratio=0.8):
    """
    scaler 仍然只拟合 baseline 输入:
      selected process + f62/f63/f64

    AR 特征是基于已经标准化后的 f63/f64 计算出来的，
    不额外拟合 scaler，避免污染原始流程。
    """
    n = len(data)
    visible_idx, batch_len, visible_len = _make_visible_indices(
        n=n,
        n_batches=n_batches,
        visible_ratio=visible_ratio,
    )

    input_cols = np.concatenate([SELECTED_COLS, TARGET_COLS])

    input_x = np.asarray(data[visible_idx][:, input_cols], dtype=np.float32)
    target_y = np.asarray(data[visible_idx][:, DL_TARGET_COLS], dtype=np.float32)

    input_scaler = Standardizer.fit(input_x)
    target_scaler = Standardizer.fit(target_y)

    return input_scaler, target_scaler, batch_len, visible_len


def weighted_loss(pred, target, loss_weights=(1.0, 1.0)):
    e = pred - target
    loss63 = torch.mean(e[..., 0] ** 2)
    loss64 = torch.mean(e[..., 1] ** 2)
    return float(loss_weights[0]) * loss63 + float(loss_weights[1]) * loss64


def _target_metrics_np(pred_np, true_np):
    out = {}

    for i, name in enumerate(DL_TARGET_NAMES):
        err = pred_np[:, i].astype(np.float64) - true_np[:, i].astype(np.float64)
        out[f"mse_{name}"] = float(np.mean(err * err))
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"bias_{name}"] = float(np.mean(err))

    out["mse_sum"] = float(sum(out[f"mse_{name}"] for name in DL_TARGET_NAMES))
    return out


def _save_ckpt(path, model, input_scaler, target_scaler, config, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pack = {
        "model": model.state_dict(),
        "input_scaler": input_scaler.state_dict(),
        "target_scaler": target_scaler.state_dict(),
        "config": config,
    }

    if extra:
        pack.update(extra)

    torch.save(pack, path)


def train_official_windows(
    data,
    labels,
    out_dir,
    n_batches=10,
    visible_ratio=0.8,
    history_len=8192,
    batch_size=1,
    epochs=30,
    lr=1e-3,
    hidden=128,
    levels=4,
    dropout=0.1,
    loss_weights=(1.0, 1.0),
    seed=42,
    device="auto",
    num_workers=0,
    near_switch_win=512,
    init_ckpt=None,
):
    """
    direct + explicit AR features 实验版。

    和 baseline direct 的区别：
      Dataset 换成 OfficialBatchWindowDatasetARFeatures
      模型仍然是 DualHeadTCNForecaster
      不做递推
      不改原 dataset.py
    """
    set_seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(data)

    input_scaler, target_scaler, batch_len, visible_len = fit_scalers(
        data=data,
        n_batches=n_batches,
        visible_ratio=visible_ratio,
    )

    pred_len = batch_len - visible_len

    if pred_len <= 0:
        raise ValueError(f"pred_len={pred_len} 非法")

    if history_len > visible_len:
        raise ValueError(
            f"history_len={history_len} 不能大于 visible_len={visible_len}"
        )

    print(
        f"[INFO] direct_ar_features windows: n={n}, n_batches={n_batches}, "
        f"batch_len={batch_len}, visible_len={visible_len}, pred_len={pred_len}",
        flush=True,
    )

    ds = OfficialBatchWindowDatasetARFeatures(
        data=data,
        labels=labels,
        history_len=history_len,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        batch_len=batch_len,
        visible_ratio=visible_ratio,
        near_switch_win=near_switch_win,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    sample0 = ds[0]
    hist_dim = int(sample0["hist"].shape[-1])
    future_dim = int(sample0["future"].shape[-1])

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    model = DualHeadTCNForecaster(
        hist_dim=hist_dim,
        future_dim=future_dim,
        hidden=hidden,
        levels=levels,
        dropout=dropout,
    ).to(device)

    loaded_from = None
    if init_ckpt:
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        loaded_from = str(init_ckpt)
        print(f"[INIT] loaded from {loaded_from}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt_cfg = {
        "mode": "official_windows",
        "model_kind": "direct_ar_features",
        "use_ar_features": True,
        "ar_feature_version": "v1_lag_roll_ewm_state",
        "n": int(n),
        "n_batches": int(n_batches),
        "visible_ratio": float(visible_ratio),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "pred_len": int(pred_len),
        "history_len": int(history_len),
        "hist_dim": int(hist_dim),
        "future_dim": int(future_dim),
        "hidden": int(hidden),
        "levels": int(levels),
        "dropout": float(dropout),
        "loss_weights": list(loss_weights),
        "target_cols": _as_list(DL_TARGET_COLS),
        "target_names": list(DL_TARGET_NAMES),
        "selected_cols": _as_list(SELECTED_COLS),
        "near_switch_win": int(near_switch_win),
        "init_ckpt": loaded_from,
    }

    history = []
    best = {"mse_sum": float("inf"), "epoch": -1}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for b in loader:
            hist = b["hist"].to(device)
            future = b["future"].to(device)
            target = b["target"].to(device)

            pred = model(hist, future)
            loss = weighted_loss(pred, target, loss_weights=loss_weights)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        sched.step()

        model.eval()

        batch_rows = []
        weighted_mse_num = {name: 0.0 for name in DL_TARGET_NAMES}
        weighted_len_sum = 0

        pred_all = []
        true_all = []

        with torch.no_grad():
            for bi in range(len(ds)):
                s = ds[bi]

                raw = model(
                    s["hist"][None].to(device),
                    s["future"][None].to(device),
                )[0].cpu().numpy()

                pred_np = target_scaler.inverse_np(raw).astype(np.float32)

                start = bi * batch_len + visible_len
                end = (bi + 1) * batch_len

                true_np = np.asarray(
                    data[start:end, DL_TARGET_COLS],
                    dtype=np.float32,
                )

                m = _target_metrics_np(pred_np, true_np)
                seg_len = int(end - start)

                for name in DL_TARGET_NAMES:
                    weighted_mse_num[name] += m[f"mse_{name}"] * seg_len

                weighted_len_sum += seg_len

                row = {
                    "batch": int(bi),
                    "start": int(start),
                    "end": int(end),
                    "length": int(seg_len),
                    **m,
                }

                batch_rows.append(row)
                pred_all.append(pred_np)
                true_all.append(true_np)

        pred_all = np.concatenate(pred_all, axis=0)
        true_all = np.concatenate(true_all, axis=0)

        metrics = {}

        for i, name in enumerate(DL_TARGET_NAMES):
            metrics[f"mse_{name}"] = float(
                weighted_mse_num[name] / max(1, weighted_len_sum)
            )

            err = pred_all[:, i].astype(np.float64) - true_all[:, i].astype(np.float64)
            metrics[f"mae_{name}"] = float(np.mean(np.abs(err)))
            metrics[f"bias_{name}"] = float(np.mean(err))

        metrics["mse_sum"] = float(
            sum(metrics[f"mse_{name}"] for name in DL_TARGET_NAMES)
        )
        metrics["epoch"] = int(ep)
        metrics["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        metrics["lr"] = float(sched.get_last_lr()[0])
        metrics["batch_metrics"] = batch_rows

        history.append(metrics)

        print(
            f"[direct_ar] ep={ep:03d} "
            f"loss={metrics['train_loss']:.6f} "
            f"sum={metrics['mse_sum']:.6f} "
            f"f63={metrics['mse_f63']:.6f} "
            f"f64={metrics['mse_f64']:.6f}",
            flush=True,
        )

        if metrics["mse_sum"] < best["mse_sum"]:
            best = dict(metrics)

            _save_ckpt(
                out_dir / "best.pt",
                model,
                input_scaler,
                target_scaler,
                ckpt_cfg,
                {
                    "best": best,
                    "mode": "official_windows",
                    "model_kind": "direct_ar_features",
                },
            )

            save_json(batch_rows, out_dir / "best_batch_metrics.json")

    save_json(
        {
            "best": best,
            "seconds": round(time.time() - t0, 1),
            "init_ckpt": loaded_from,
            "history": history,
            "config": ckpt_cfg,
        },
        out_dir / "summary.json",
    )

    return best