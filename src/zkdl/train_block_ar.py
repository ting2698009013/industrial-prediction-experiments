from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SELECTED_COLS, TARGET_COLS, DL_TARGET_COLS, DL_TARGET_NAMES
from .dataset import OfficialBatchWindowDataset
from .model_block_ar import DualHeadTCNForecaster, BlockARDualHeadTCNForecaster
from .scaler import Standardizer
from .utils import set_seed, save_json


def _load_init_weights(model, init_ckpt):
    if not init_ckpt:
        return None

    p = Path(init_ckpt)
    if not p.exists():
        raise FileNotFoundError(f"init_ckpt 不存在: {init_ckpt}")

    ckpt = torch.load(str(p), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state, strict=True)
    print(f"[INIT] loaded model weights from {p}", flush=True)
    return str(p)


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
    只使用每个 batch 前 visible_ratio 的可见段拟合 scaler，
    避免使用待预测段目标信息。

    input_scaler 拟合：
      selected process features + f62/f63/f64 历史目标

    target_scaler 拟合：
      f63/f64
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
    """
    pred/target shape = (B, T, 2)
    顺序为 f63, f64
    """
    e = pred - target
    loss63 = torch.mean(e[..., 0] ** 2)
    loss64 = torch.mean(e[..., 1] ** 2)
    return float(loss_weights[0]) * loss63 + float(loss_weights[1]) * loss64


def _target_metrics_np(pred_np, true_np):
    """
    pred_np/true_np shape = (T, 2)
    顺序为 f63, f64
    """
    out = {}

    for i, name in enumerate(DL_TARGET_NAMES):
        err = pred_np[:, i].astype(np.float64) - true_np[:, i].astype(np.float64)
        out[f"mse_{name}"] = float(np.mean(err * err))
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"bias_{name}"] = float(np.mean(err))

    out["mse_sum"] = float(sum(out[f"mse_{name}"] for name in DL_TARGET_NAMES))
    return out


def _build_model(
    model_kind,
    hist_dim,
    future_dim,
    hidden,
    levels,
    dropout,
    block_len,
    detach_ar,
):
    """
    model_kind:
      direct   : 当前 direct dual-head baseline
      block_ar : block autoregressive dual-head
    """

    # hist 的结构来自 dataset.build_input_matrix:
    # selected process features + f62/f63/f64 + mode onehot/switch features
    #
    # 其中：
    #   f62 位于 len(SELECTED_COLS) + 0
    #   f63 位于 len(SELECTED_COLS) + 1
    #   f64 位于 len(SELECTED_COLS) + 2
    #
    # block AR 初始状态使用 hist 最后一个可见点的 f63/f64。
    ar_target_indices = (
        int(len(SELECTED_COLS) + 1),
        int(len(SELECTED_COLS) + 2),
    )

    if model_kind == "direct":
        model = DualHeadTCNForecaster(
            hist_dim=hist_dim,
            future_dim=future_dim,
            hidden=hidden,
            levels=levels,
            dropout=dropout,
        )

    elif model_kind == "block_ar":
        model = BlockARDualHeadTCNForecaster(
            hist_dim=hist_dim,
            future_dim=future_dim,
            hidden=hidden,
            levels=levels,
            dropout=dropout,
            block_len=block_len,
            ar_target_indices=ar_target_indices,
            detach_ar=detach_ar,
        )

    else:
        raise ValueError(f"unknown model_kind: {model_kind}")

    return model, ar_target_indices


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
    model_kind="block_ar",
    block_len=256,
    detach_ar=True,
):
    """
    全量 official-window 训练：

    - 完整训练集按时间平均切成 n_batches 个 batch
    - 每个 batch 前 visible_ratio 作为可见段
    - 每个 batch 后 1-visible_ratio 作为预测/监督段
    - 输入里每个 batch 后 20% 的 f62/f63/f64 由 OfficialBatchWindowDataset 置 0
    - MSE 只在每个 batch 的预测段计算
    - 最终 MSE 为各 batch 按长度加权平均
    - batch 等长时等价于 batch MSE 平均

    model_kind:
      direct:
        普通 direct dual-head，对照 baseline

      block_ar:
        block autoregressive dual-head。
        future 按 block_len 分块，每个 block 使用上一个 block 的最后预测值作为 AR 状态。
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
        raise ValueError(f"pred_len={pred_len} 非法，请检查 visible_ratio={visible_ratio}")

    if history_len > visible_len:
        raise ValueError(
            f"history_len={history_len} 不能大于 visible_len={visible_len}。"
            f"请减小 history_len 或 visible_ratio。"
        )

    print(
        f"[INFO] official windows: n={n}, n_batches={n_batches}, "
        f"batch_len={batch_len}, visible_len={visible_len}, pred_len={pred_len}",
        flush=True,
    )

    print(
        f"[INFO] model_kind={model_kind}, block_len={block_len}, detach_ar={detach_ar}",
        flush=True,
    )

    ds = OfficialBatchWindowDataset(
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

    model, ar_target_indices = _build_model(
        model_kind=model_kind,
        hist_dim=hist_dim,
        future_dim=future_dim,
        hidden=hidden,
        levels=levels,
        dropout=dropout,
        block_len=block_len,
        detach_ar=detach_ar,
    )

    model = model.to(device)

    loaded_from = _load_init_weights(model, init_ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt_cfg = {
        "mode": "official_windows",
        "model_kind": str(model_kind),
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
        "target_cols": DL_TARGET_COLS.tolist(),
        "target_names": list(DL_TARGET_NAMES),
        "selected_cols": SELECTED_COLS.tolist(),
        "near_switch_win": int(near_switch_win),
        "init_ckpt": loaded_from,
        "block_len": int(block_len),
        "detach_ar": bool(detach_ar),
        "ar_target_indices": list(ar_target_indices),
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
            f"[final/{model_kind}] ep={ep:03d} "
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
                    "model_kind": str(model_kind),
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