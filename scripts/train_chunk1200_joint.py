#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkdl.config import SELECTED_COLS, TARGET_COLS, DL_TARGET_COLS, DL_TARGET_NAMES
from zkdl.dataset import build_input_matrix, time_features
from zkdl.model import DualHeadTCNForecaster
from zkdl.scaler import Standardizer
from zkdl.utils import set_seed


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=default),
        encoding="utf-8",
    )


def parse_weights(s):
    parts = str(s).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("--loss-weights 需要两个数，例如 1.0,1.0")
    return float(parts[0]), float(parts[1])


def make_scaler_indices(n, n_batches, visible_ratio, visible_only):
    if n % n_batches != 0:
        raise ValueError(f"len(data)={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))

    idx_parts = []
    for bi in range(n_batches):
        s = bi * batch_len
        e = s + (visible_len if visible_only else batch_len)
        idx_parts.append(np.arange(s, e, dtype=np.int64))

    return np.concatenate(idx_parts), batch_len, visible_len


def fit_scalers(data, n_batches, visible_ratio, visible_only):
    idx, batch_len, visible_len = make_scaler_indices(
        n=len(data),
        n_batches=n_batches,
        visible_ratio=visible_ratio,
        visible_only=visible_only,
    )

    input_cols = np.concatenate([SELECTED_COLS, TARGET_COLS])

    x = np.asarray(data[idx][:, input_cols], dtype=np.float32)
    y = np.asarray(data[idx][:, DL_TARGET_COLS], dtype=np.float32)

    input_scaler = Standardizer.fit(x)
    target_scaler = Standardizer.fit(y)

    return input_scaler, target_scaler, batch_len, visible_len


class ChunkWindowDataset(Dataset):
    """
    随机 chunk 监督样本。

    每个样本：
      hist   = [cut-history_len, cut)
      future = [cut, cut+chunk_len)
      target = f63/f64 at [cut, cut+chunk_len)

    注意：
      future 输入不包含未来 f63/f64。
      hist 中只包含 cut 之前的 f62/f63/f64 历史。
    """

    def __init__(
        self,
        data,
        labels,
        input_scaler,
        target_scaler,
        n_batches=10,
        visible_ratio=0.8,
        history_len=16384,
        chunk_len=1200,
        samples_per_batch=64,
        train_visible_only=False,
        selected_cols=SELECTED_COLS,
        near_switch_win=512,
    ):
        self.data = data
        self.labels = labels
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.n_batches = int(n_batches)
        self.visible_ratio = float(visible_ratio)
        self.history_len = int(history_len)
        self.chunk_len = int(chunk_len)
        self.samples_per_batch = int(samples_per_batch)
        self.train_visible_only = bool(train_visible_only)
        self.selected_cols = selected_cols
        self.near_switch_win = int(near_switch_win)

        n = len(data)
        if n % self.n_batches != 0:
            raise ValueError(f"len(data)={n} 不能被 n_batches={self.n_batches} 整除")

        self.batch_len = n // self.n_batches
        self.visible_len = int(round(self.batch_len * self.visible_ratio))

        self.hist_all, self.fut_all, self.targets_all = build_input_matrix(
            data,
            labels,
            selected_cols=self.selected_cols,
            zero_future_target_mask=None,
            near_switch_win=self.near_switch_win,
        )

        self.n_proc = len(self.selected_cols)

        self.hist_all = self.hist_all.copy()
        self.fut_all = self.fut_all.copy()

        self.hist_all[:, : self.n_proc + 3] = self.input_scaler.transform_np(
            self.hist_all[:, : self.n_proc + 3]
        )

        dummy = np.concatenate(
            [
                self.fut_all[:, : self.n_proc],
                np.zeros((len(self.fut_all), 3), dtype=np.float32),
            ],
            axis=1,
        )
        self.fut_all[:, : self.n_proc] = self.input_scaler.transform_np(dummy)[
            :, : self.n_proc
        ]

        self.targets_scaled = self.target_scaler.transform_np(self.targets_all)

        self.valid_batches = []
        for bi in range(self.n_batches):
            s = bi * self.batch_len
            if self.train_visible_only:
                usable_end = s + self.visible_len
            else:
                usable_end = s + self.batch_len

            min_cut = s + self.history_len
            max_cut = usable_end - self.chunk_len

            if max_cut >= min_cut:
                self.valid_batches.append((bi, s, min_cut, max_cut, usable_end))

        if not self.valid_batches:
            raise ValueError(
                "没有可采样窗口，请检查 history_len/chunk_len/visible_len"
            )

    def __len__(self):
        return len(self.valid_batches) * self.samples_per_batch

    def __getitem__(self, idx):
        j = int(idx) % len(self.valid_batches)
        bi, s, min_cut, max_cut, usable_end = self.valid_batches[j]

        cut = np.random.randint(min_cut, max_cut + 1)

        hs = cut - self.history_len
        he = cut
        fs = cut
        fe = cut + self.chunk_len

        hist = self.hist_all[hs:he]
        fut = np.concatenate(
            [self.fut_all[fs:fe], time_features(self.chunk_len)],
            axis=1,
        ).astype(np.float32)
        y = self.targets_scaled[fs:fe].astype(np.float32)

        return {
            "hist": torch.from_numpy(hist),
            "future": torch.from_numpy(fut),
            "target": torch.from_numpy(y),
            "batch": int(bi),
            "cut": int(cut),
        }


def weighted_mse_loss(pred, target, loss_weights):
    w63, w64 = loss_weights
    loss63 = torch.mean((pred[..., 0] - target[..., 0]) ** 2)
    loss64 = torch.mean((pred[..., 1] - target[..., 1]) ** 2)
    return w63 * loss63 + w64 * loss64, loss63, loss64


def evaluate_fixed_chunks(
    model,
    data,
    labels,
    input_scaler,
    target_scaler,
    n_batches,
    visible_ratio,
    history_len,
    chunk_len,
    eval_visible_only,
    device,
    near_switch_win=512,
):
    """
    用固定窗口评估：
      每个 batch 取 usable_end - chunk_len 作为 cut。
    """
    model.eval()

    n = len(data)
    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))

    hist_all, fut_all, targets_all = build_input_matrix(
        data,
        labels,
        selected_cols=SELECTED_COLS,
        zero_future_target_mask=None,
        near_switch_win=near_switch_win,
    )

    n_proc = len(SELECTED_COLS)

    hist_all = hist_all.copy()
    fut_all = fut_all.copy()

    hist_all[:, : n_proc + 3] = input_scaler.transform_np(
        hist_all[:, : n_proc + 3]
    )

    dummy = np.concatenate(
        [
            fut_all[:, :n_proc],
            np.zeros((len(fut_all), 3), dtype=np.float32),
        ],
        axis=1,
    )
    fut_all[:, :n_proc] = input_scaler.transform_np(dummy)[:, :n_proc]

    rows = []
    pred_all = []
    true_all = []

    with torch.no_grad():
        for bi in range(n_batches):
            s = bi * batch_len
            usable_end = s + (visible_len if eval_visible_only else batch_len)

            cut = usable_end - chunk_len
            if cut - history_len < s:
                continue

            hist = hist_all[cut - history_len : cut].astype(np.float32)
            fut = np.concatenate(
                [fut_all[cut : cut + chunk_len], time_features(chunk_len)],
                axis=1,
            ).astype(np.float32)

            hist_t = torch.from_numpy(hist[None]).to(device)
            fut_t = torch.from_numpy(fut[None]).to(device)

            raw = model(hist_t, fut_t)[0].cpu().numpy().astype(np.float32)
            pred = target_scaler.inverse_np(raw).astype(np.float32)
            true = np.asarray(
                data[cut : cut + chunk_len, DL_TARGET_COLS],
                dtype=np.float32,
            )

            err = pred.astype(np.float64) - true.astype(np.float64)

            row = {
                "batch": int(bi),
                "cut": int(cut),
                "length": int(chunk_len),
                "mse_f63": float(np.mean(err[:, 0] ** 2)),
                "mse_f64": float(np.mean(err[:, 1] ** 2)),
                "mae_f63": float(np.mean(np.abs(err[:, 0]))),
                "mae_f64": float(np.mean(np.abs(err[:, 1]))),
                "bias_f63": float(np.mean(err[:, 0])),
                "bias_f64": float(np.mean(err[:, 1])),
            }
            row["mse_sum"] = row["mse_f63"] + row["mse_f64"]

            rows.append(row)
            pred_all.append(pred)
            true_all.append(true)

    if not rows:
        return {
            "mse_f63": float("inf"),
            "mse_f64": float("inf"),
            "mse_sum": float("inf"),
            "batch_metrics": [],
        }

    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)
    err = pred_all.astype(np.float64) - true_all.astype(np.float64)

    summary = {
        "mse_f63": float(np.mean(err[:, 0] ** 2)),
        "mse_f64": float(np.mean(err[:, 1] ** 2)),
        "mae_f63": float(np.mean(np.abs(err[:, 0]))),
        "mae_f64": float(np.mean(np.abs(err[:, 1]))),
        "bias_f63": float(np.mean(err[:, 0])),
        "bias_f64": float(np.mean(err[:, 1])),
        "batch_metrics": rows,
    }
    summary["mse_sum"] = summary["mse_f63"] + summary["mse_f64"]

    return summary


def save_ckpt(path, model, input_scaler, target_scaler, config, best):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "input_scaler": input_scaler.state_dict(),
            "target_scaler": target_scaler.state_dict(),
            "config": config,
            "best": best,
            "model_kind": "chunked_joint_tcn",
        },
        path,
    )


def train_chunk_model(
    data_path,
    labels_path,
    out_dir,
    n_batches=10,
    visible_ratio=0.8,
    history_len=16384,
    chunk_len=1200,
    samples_per_batch=64,
    train_visible_only=False,
    scaler_visible_only=False,
    batch_size=8,
    epochs=200,
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
    use_init_scalers=True,
):
    set_seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if data.ndim != 2 or data.shape[1] != 65:
        raise ValueError(f"data 应该是 (N,65)，现在是 {data.shape}")
    if len(labels) != len(data):
        raise ValueError(f"labels length {len(labels)} != data length {len(data)}")

    ckpt = None
    if init_ckpt:
        ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)

    if init_ckpt and use_init_scalers:
        input_scaler = Standardizer.from_state_dict(ckpt["input_scaler"])
        target_scaler = Standardizer.from_state_dict(ckpt["target_scaler"])
        _, batch_len, visible_len = make_scaler_indices(
            len(data), n_batches, visible_ratio, visible_only=True
        )
        print(f"[INIT] use scalers from {init_ckpt}")
    else:
        input_scaler, target_scaler, batch_len, visible_len = fit_scalers(
            data=data,
            n_batches=n_batches,
            visible_ratio=visible_ratio,
            visible_only=scaler_visible_only,
        )
        print(
            f"[SCALER] fit on {'visible only' if scaler_visible_only else 'full available data'}"
        )

    if history_len + chunk_len > (visible_len if train_visible_only else batch_len):
        raise ValueError(
            f"history_len + chunk_len 太大: {history_len}+{chunk_len}, "
            f"usable={visible_len if train_visible_only else batch_len}"
        )

    ds = ChunkWindowDataset(
        data=data,
        labels=labels,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        n_batches=n_batches,
        visible_ratio=visible_ratio,
        history_len=history_len,
        chunk_len=chunk_len,
        samples_per_batch=samples_per_batch,
        train_visible_only=train_visible_only,
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

    if init_ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[INIT] loaded model weights from {init_ckpt}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, epochs),
    )

    cfg = {
        "model_kind": "chunked_joint_tcn",
        "data": str(data_path),
        "labels": str(labels_path),
        "n": int(len(data)),
        "n_batches": int(n_batches),
        "visible_ratio": float(visible_ratio),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "history_len": int(history_len),
        "chunk_len": int(chunk_len),
        "samples_per_batch": int(samples_per_batch),
        "train_visible_only": bool(train_visible_only),
        "scaler_visible_only": bool(scaler_visible_only),
        "hist_dim": int(hist_dim),
        "future_dim": int(future_dim),
        "hidden": int(hidden),
        "levels": int(levels),
        "dropout": float(dropout),
        "loss_weights": [float(loss_weights[0]), float(loss_weights[1])],
        "target_cols": [int(x) for x in DL_TARGET_COLS],
        "target_names": list(DL_TARGET_NAMES),
        "selected_cols": [int(x) for x in SELECTED_COLS],
        "near_switch_win": int(near_switch_win),
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
    }

    save_json(cfg, out_dir / "config.json")

    best = {
        "epoch": -1,
        "mse_sum": float("inf"),
        "mse_f63": float("inf"),
        "mse_f64": float("inf"),
    }
    history = []
    t0 = time.time()

    print(
        f"[INFO] n={len(data)}, batch_len={batch_len}, visible_len={visible_len}, "
        f"history_len={history_len}, chunk_len={chunk_len}, "
        f"train_visible_only={train_visible_only}, samples_per_batch={samples_per_batch}",
        flush=True,
    )
    print(f"[INFO] hist_dim={hist_dim}, future_dim={future_dim}, device={device}")

    for ep in range(1, epochs + 1):
        model.train()

        losses = []
        losses63 = []
        losses64 = []

        for batch in loader:
            hist = batch["hist"].to(device)
            fut = batch["future"].to(device)
            target = batch["target"].to(device)

            pred = model(hist, fut)
            loss, loss63, loss64 = weighted_mse_loss(pred, target, loss_weights)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))
            losses63.append(float(loss63.item()))
            losses64.append(float(loss64.item()))

        sched.step()

        metrics = evaluate_fixed_chunks(
            model=model,
            data=data,
            labels=labels,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            n_batches=n_batches,
            visible_ratio=visible_ratio,
            history_len=history_len,
            chunk_len=chunk_len,
            eval_visible_only=train_visible_only,
            device=device,
            near_switch_win=near_switch_win,
        )

        metrics["epoch"] = int(ep)
        metrics["train_loss"] = float(np.mean(losses))
        metrics["train_loss63"] = float(np.mean(losses63))
        metrics["train_loss64"] = float(np.mean(losses64))
        metrics["lr"] = float(sched.get_last_lr()[0])

        history.append(metrics)

        print(
            f"[chunk1200] ep={ep:03d} "
            f"loss={metrics['train_loss']:.6f} "
            f"eval_f63={metrics['mse_f63']:.6f} "
            f"eval_f64={metrics['mse_f64']:.6f} "
            f"eval_sum={metrics['mse_sum']:.6f}",
            flush=True,
        )

        if metrics["mse_sum"] < best["mse_sum"]:
            best = dict(metrics)
            save_ckpt(
                out_dir / "best.pt",
                model=model,
                input_scaler=input_scaler,
                target_scaler=target_scaler,
                config=cfg,
                best=best,
            )
            save_json(metrics["batch_metrics"], out_dir / "best_batch_metrics.json")
            print(
                f"[BEST] ep={ep:03d} mse_sum={metrics['mse_sum']:.6f}",
                flush=True,
            )

    summary = {
        "config": cfg,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }

    save_json(summary, out_dir / "run_summary.json")
    save_json(summary, out_dir / "summary.json")

    print("[DONE] best:", best)
    print("[OK] saved:", out_dir / "best.pt")

    return best


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--history-len", type=int, default=16384)
    p.add_argument("--chunk-len", type=int, default=1200)
    p.add_argument("--samples-per-batch", type=int, default=64)

    p.add_argument("--train-visible-only", action="store_true")
    p.add_argument("--scaler-visible-only", action="store_true")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--loss-weights", default="1.0,1.0")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--near-switch-win", type=int, default=512)

    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--no-init-scalers", action="store_true")

    args = p.parse_args()

    train_chunk_model(
        data_path=args.data,
        labels_path=args.labels,
        out_dir=args.out,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        history_len=args.history_len,
        chunk_len=args.chunk_len,
        samples_per_batch=args.samples_per_batch,
        train_visible_only=args.train_visible_only,
        scaler_visible_only=args.scaler_visible_only,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        levels=args.levels,
        dropout=args.dropout,
        loss_weights=parse_weights(args.loss_weights),
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        near_switch_win=args.near_switch_win,
        init_ckpt=args.init_ckpt,
        use_init_scalers=not args.no_init_scalers,
    )


if __name__ == "__main__":
    main()