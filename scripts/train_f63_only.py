#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkdl.config import SELECTED_COLS, TARGET_COLS
from zkdl.dataset import build_input_matrix, make_question_style_zero_mask, time_features
from zkdl.model import TCNEncoder
from zkdl.scaler import Standardizer
from zkdl.utils import set_seed


F63_COL = 63


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


class SingleHeadTCNForecaster(nn.Module):
    """
    只预测 f63。
    输入：
      hist:   (B, history_len, hist_dim)
      future: (B, pred_len, future_dim)
    输出：
      y63:    (B, pred_len, 1)
    """

    def __init__(self, hist_dim, future_dim, hidden=128, levels=4, dropout=0.1):
        super().__init__()
        self.hist_encoder = TCNEncoder(hist_dim, hidden, levels, dropout=dropout)
        self.future_encoder = TCNEncoder(future_dim, hidden, levels, dropout=dropout)

        self.context_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, hist, future):
        hz = self.hist_encoder(hist)
        ctx = self.context_proj(torch.cat([hz[:, -1], hz.mean(dim=1)], dim=-1))

        fz = self.future_encoder(future)
        ctx = ctx[:, None, :].expand(-1, fz.shape[1], -1)

        h = self.shared(torch.cat([fz, ctx], dim=-1))
        y = self.head(h)
        return y


class F63OfficialDataset(Dataset):
    """
    官方窗口训练：
      每个 batch 前 visible_len 可见，后 pred_len 监督。
    """

    def __init__(
        self,
        data,
        labels,
        history_len,
        input_scaler,
        f63_scaler,
        batch_len,
        visible_ratio=0.8,
        near_switch_win=512,
    ):
        self.data = data
        self.labels = labels
        self.history_len = int(history_len)
        self.batch_len = int(batch_len)
        self.visible_len = int(round(batch_len * visible_ratio))
        self.pred_len = self.batch_len - self.visible_len
        self.n_batches = len(data) // self.batch_len
        self.input_scaler = input_scaler
        self.f63_scaler = f63_scaler

        zmask = make_question_style_zero_mask(
            len(data),
            batch_len=self.batch_len,
            visible_ratio=visible_ratio,
        )

        self.hist_all, self.fut_all, _ = build_input_matrix(
            data,
            labels,
            selected_cols=SELECTED_COLS,
            zero_future_target_mask=zmask,
            near_switch_win=near_switch_win,
        )

        self.n_proc = len(SELECTED_COLS)

        self.hist_all = self.hist_all.copy()
        self.fut_all = self.fut_all.copy()

        # hist 前 n_proc+3 列：process + f62/f63/f64 history
        self.hist_all[:, : self.n_proc + 3] = input_scaler.transform_np(
            self.hist_all[:, : self.n_proc + 3]
        )

        # future 只有 process + mode，需要用 dummy target 对齐 input_scaler
        dummy = np.concatenate(
            [
                self.fut_all[:, : self.n_proc],
                np.zeros((len(self.fut_all), 3), dtype=np.float32),
            ],
            axis=1,
        )
        self.fut_all[:, : self.n_proc] = input_scaler.transform_np(dummy)[:, : self.n_proc]

        self.f63_scaled = f63_scaler.transform_np(
            np.asarray(data[:, [F63_COL]], dtype=np.float32)
        )

        if self.history_len > self.visible_len:
            raise ValueError(
                f"history_len={self.history_len} > visible_len={self.visible_len}"
            )

    def __len__(self):
        return self.n_batches

    def __getitem__(self, bi):
        s = int(bi) * self.batch_len
        cut = s + self.visible_len
        end = s + self.batch_len
        hs = cut - self.history_len

        hist = self.hist_all[hs:cut]
        fut = np.concatenate(
            [self.fut_all[cut:end], time_features(end - cut)],
            axis=1,
        ).astype(np.float32)

        y = self.f63_scaled[cut:end].astype(np.float32)

        return {
            "hist": torch.from_numpy(hist),
            "future": torch.from_numpy(fut),
            "target": torch.from_numpy(y),
            "batch": int(bi),
        }


def make_visible_indices(n, n_batches, visible_ratio):
    if n % n_batches != 0:
        raise ValueError(f"len(data)={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    idx = []
    for bi in range(n_batches):
        s = bi * batch_len
        idx.append(np.arange(s, s + visible_len, dtype=np.int64))

    return np.concatenate(idx), batch_len, visible_len, pred_len


def fit_scalers(data, n_batches, visible_ratio):
    visible_idx, batch_len, visible_len, pred_len = make_visible_indices(
        len(data),
        n_batches=n_batches,
        visible_ratio=visible_ratio,
    )

    input_cols = np.concatenate([SELECTED_COLS, TARGET_COLS])
    x = np.asarray(data[visible_idx][:, input_cols], dtype=np.float32)
    y63 = np.asarray(data[visible_idx][:, [F63_COL]], dtype=np.float32)

    input_scaler = Standardizer.fit(x)
    f63_scaler = Standardizer.fit(y63)

    return input_scaler, f63_scaler, batch_len, visible_len, pred_len


def inverse_f63(f63_scaler, arr):
    return f63_scaler.inverse_np(arr.reshape(-1, 1)).reshape(arr.shape)


def evaluate(model, ds, data, f63_scaler, batch_len, visible_len, device):
    model.eval()

    pred_all = []
    true_all = []
    batch_metrics = []

    with torch.no_grad():
        for bi in range(len(ds)):
            sample = ds[bi]

            hist = sample["hist"][None].to(device)
            future = sample["future"][None].to(device)

            raw = model(hist, future)[0].cpu().numpy().astype(np.float32)
            pred = inverse_f63(f63_scaler, raw[:, 0])

            s = bi * batch_len + visible_len
            e = (bi + 1) * batch_len
            true = np.asarray(data[s:e, F63_COL], dtype=np.float32)

            err = pred.astype(np.float64) - true.astype(np.float64)

            row = {
                "batch": int(bi),
                "start": int(s),
                "end": int(e),
                "length": int(e - s),
                "mse_f63": float(np.mean(err * err)),
                "mae_f63": float(np.mean(np.abs(err))),
                "bias_f63": float(np.mean(err)),
            }

            batch_metrics.append(row)
            pred_all.append(pred)
            true_all.append(true)

    pred_all = np.concatenate(pred_all)
    true_all = np.concatenate(true_all)
    err = pred_all.astype(np.float64) - true_all.astype(np.float64)

    return {
        "mse_f63": float(np.mean(err * err)),
        "mae_f63": float(np.mean(np.abs(err))),
        "bias_f63": float(np.mean(err)),
        "batch_metrics": batch_metrics,
    }


def save_ckpt(path, model, input_scaler, f63_scaler, config, best):
    pack = {
        "model": model.state_dict(),
        "input_scaler": input_scaler.state_dict(),
        "f63_scaler": f63_scaler.state_dict(),
        "config": config,
        "best": best,
        "model_kind": "f63_single",
    }
    torch.save(pack, path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--history-len", type=int, default=16384)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--near-switch-win", type=int, default=512)

    args = p.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")

    if data.ndim != 2 or data.shape[1] != 65:
        raise ValueError(f"data 应该是 (N,65)，现在是 {data.shape}")

    if len(labels) != len(data):
        raise ValueError(f"labels length {len(labels)} != data length {len(data)}")

    input_scaler, f63_scaler, batch_len, visible_len, pred_len = fit_scalers(
        data,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
    )

    ds = F63OfficialDataset(
        data=data,
        labels=labels,
        history_len=args.history_len,
        input_scaler=input_scaler,
        f63_scaler=f63_scaler,
        batch_len=batch_len,
        visible_ratio=args.visible_ratio,
        near_switch_win=args.near_switch_win,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    sample0 = ds[0]
    hist_dim = int(sample0["hist"].shape[-1])
    future_dim = int(sample0["future"].shape[-1])

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    device = torch.device(device)

    model = SingleHeadTCNForecaster(
        hist_dim=hist_dim,
        future_dim=future_dim,
        hidden=args.hidden,
        levels=args.levels,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.epochs),
    )

    cfg = {
        "model_kind": "f63_single",
        "n": int(len(data)),
        "n_batches": int(args.n_batches),
        "visible_ratio": float(args.visible_ratio),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "pred_len": int(pred_len),
        "history_len": int(args.history_len),
        "hist_dim": int(hist_dim),
        "future_dim": int(future_dim),
        "hidden": int(args.hidden),
        "levels": int(args.levels),
        "dropout": float(args.dropout),
        "selected_cols": [int(x) for x in SELECTED_COLS],
        "target": "f63",
        "target_col": 63,
        "near_switch_win": int(args.near_switch_win),
    }

    best = {"mse_f63": float("inf"), "epoch": -1}
    history = []
    t0 = time.time()

    print(
        f"[INFO] f63-single: n={len(data)}, batch_len={batch_len}, "
        f"visible_len={visible_len}, pred_len={pred_len}, "
        f"history_len={args.history_len}, hist_dim={hist_dim}, future_dim={future_dim}",
        flush=True,
    )

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in loader:
            hist = batch["hist"].to(device)
            future = batch["future"].to(device)
            target = batch["target"].to(device)

            pred = model(hist, future)
            loss = torch.mean((pred - target) ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))

        sched.step()

        metrics = evaluate(
            model=model,
            ds=ds,
            data=data,
            f63_scaler=f63_scaler,
            batch_len=batch_len,
            visible_len=visible_len,
            device=device,
        )

        metrics["epoch"] = int(ep)
        metrics["train_loss"] = float(np.mean(losses))
        metrics["lr"] = float(sched.get_last_lr()[0])
        history.append(metrics)

        print(
            f"[f63-single] ep={ep:03d} "
            f"loss={metrics['train_loss']:.6f} "
            f"mse_f63={metrics['mse_f63']:.6f} "
            f"mae_f63={metrics['mae_f63']:.6f}",
            flush=True,
        )

        if metrics["mse_f63"] < best["mse_f63"]:
            best = dict(metrics)

            save_ckpt(
                out_dir / "best.pt",
                model=model,
                input_scaler=input_scaler,
                f63_scaler=f63_scaler,
                config=cfg,
                best=best,
            )

            save_json(metrics["batch_metrics"], out_dir / "best_batch_metrics.json")

            print(
                f"[BEST] ep={ep:03d} mse_f63={metrics['mse_f63']:.6f}",
                flush=True,
            )

    summary = {
        "data": args.data,
        "labels": args.labels,
        "out": args.out,
        "model_kind": "f63_single",
        "best": best,
        "history_len": args.history_len,
        "hidden": args.hidden,
        "levels": args.levels,
        "dropout": args.dropout,
        "n_batches": args.n_batches,
        "visible_ratio": args.visible_ratio,
        "batch_len": batch_len,
        "visible_len": visible_len,
        "pred_len": pred_len,
        "epochs": args.epochs,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }

    save_json(summary, out_dir / "run_summary.json")
    save_json(summary, out_dir / "summary.json")

    print("[DONE] best mse_f63:", best["mse_f63"])
    print("[DONE] best epoch:", best["epoch"])
    print("[OK] saved:", out_dir / "best.pt")


if __name__ == "__main__":
    main()