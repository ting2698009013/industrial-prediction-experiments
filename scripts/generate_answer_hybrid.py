#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkdl.config import SELECTED_COLS, TARGET_COLS
from zkdl.dataset import build_input_matrix, make_question_style_zero_mask, time_features
from zkdl.model import TCNEncoder
from zkdl.scaler import Standardizer

from zkmm.features_direct import build_prediction_features_for_batch
from zkmm.config import SELECTED_COLS as ML_SELECTED_COLS


class SingleHeadTCNForecaster(nn.Module):
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
        return self.head(h)


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


def get_target_bundle(bundle, key):
    if key in bundle:
        return bundle[key]
    if "targets" in bundle and key in bundle["targets"]:
        return bundle["targets"][key]
    raise KeyError(f"bundle 里找不到 {key}")


def load_f63_single(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt.get("model_kind") != "f63_single":
        raise ValueError(
            f"这个 ckpt 不是 f63_single，model_kind={ckpt.get('model_kind')}"
        )

    cfg = ckpt["config"]

    model = SingleHeadTCNForecaster(
        hist_dim=int(cfg["hist_dim"]),
        future_dim=int(cfg["future_dim"]),
        hidden=int(cfg["hidden"]),
        levels=int(cfg["levels"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    input_scaler = Standardizer.from_state_dict(ckpt["input_scaler"])
    f63_scaler = Standardizer.from_state_dict(ckpt["f63_scaler"])

    return model, input_scaler, f63_scaler, cfg, ckpt.get("best", {})


def predict_f63_single(
    question,
    labels,
    ckpt_path,
    n_batches,
    visible_ratio,
    device,
):
    model, input_scaler, f63_scaler, cfg, best = load_f63_single(ckpt_path, device)

    n = len(question)
    if n % n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len
    history_len = int(cfg["history_len"])

    if history_len > visible_len:
        raise ValueError(
            f"history_len={history_len} > visible_len={visible_len}"
        )

    zmask = make_question_style_zero_mask(
        n,
        batch_len=batch_len,
        visible_ratio=visible_ratio,
    )

    hist_all, fut_all, _ = build_input_matrix(
        question,
        labels,
        selected_cols=SELECTED_COLS,
        zero_future_target_mask=zmask,
        near_switch_win=int(cfg.get("near_switch_win", 512)),
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

    pred63 = np.zeros((n_batches, pred_len), dtype=np.float32)

    with torch.no_grad():
        for bi in range(n_batches):
            s = bi * batch_len
            cut = s + visible_len
            end = s + batch_len
            hs = cut - history_len

            hist = hist_all[hs:cut].astype(np.float32)
            fut = np.concatenate(
                [fut_all[cut:end], time_features(pred_len)],
                axis=1,
            ).astype(np.float32)

            hist_t = torch.from_numpy(hist[None]).to(device)
            fut_t = torch.from_numpy(fut[None]).to(device)

            raw = model(hist_t, fut_t)[0].cpu().numpy().astype(np.float32)
            inv = f63_scaler.inverse_np(raw).astype(np.float32)

            pred63[bi] = inv[:, 0]

            print(
                f"[F63-DL] batch={bi}, "
                f"min={float(pred63[bi].min()):.6f}, "
                f"max={float(pred63[bi].max()):.6f}, "
                f"mean={float(pred63[bi].mean()):.6f}",
                flush=True,
            )

    return pred63, {
        "ckpt": str(ckpt_path),
        "best": best,
        "cfg": cfg,
    }


def predict_ml_target(
    question,
    labels,
    bundle_path,
    target_index,
    n_batches,
    visible_ratio,
):
    bundle = joblib.load(bundle_path)
    key = f"target_{target_index}"
    tb = get_target_bundle(bundle, key)

    model = tb["global_model"]

    n = len(question)
    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    if batch_len != 60000 or visible_len != 48000 or pred_len != 12000:
        raise ValueError(
            "当前 ML features_direct.py 使用固定 VISIBLE_LEN=48000, PRED_LEN=12000。"
            f"现在 batch_len={batch_len}, visible_len={visible_len}, pred_len={pred_len}。"
            "官方验证集是 60000 时可以用。"
        )

    pred = np.zeros((n_batches, pred_len), dtype=np.float32)

    for bi in range(n_batches):
        bs = bi * batch_len
        be = bs + batch_len

        batch = np.asarray(question[bs:be], dtype=np.float32)
        labels_batch = np.asarray(labels[bs:be])

        X, _ = build_prediction_features_for_batch(
            batch=batch,
            target_index=target_index,
            labels_batch=labels_batch,
            selected_cols=ML_SELECTED_COLS,
            add_mode=True,
            add_switch=tb.get("add_switch", True),
            cross=tb.get("cross", "core"),
            f63_state=tb.get("f63_state", "base"),
        )

        p = model.predict(X).astype(np.float32)

        if target_index == 0:
            if "f62_res_q001" in tb and "f62_res_q999" in tb:
                p = np.clip(
                    p,
                    float(tb["f62_res_q001"]),
                    float(tb["f62_res_q999"]),
                )
            p = batch[visible_len : visible_len + pred_len, 1].astype(np.float32) + p

        pred[bi] = p

        print(
            f"[ML target_{target_index}] batch={bi}, "
            f"min={float(p.min()):.6f}, "
            f"max={float(p.max()):.6f}, "
            f"mean={float(p.mean()):.6f}",
            flush=True,
        )

    return pred, {
        "bundle": str(bundle_path),
        "target_index": int(target_index),
        "meta": {k: v for k, v in tb.items() if k != "global_model"},
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--question", required=True, help="2D question, shape=(600000,65)")
    p.add_argument("--labels", required=True, help="mode labels, shape=(600000,)")

    p.add_argument("--f62-bundle", required=True)
    p.add_argument("--f63-ckpt", required=True)
    p.add_argument("--f64-bundle", required=True)

    p.add_argument("--out", required=True)
    p.add_argument("--meta-out", default=None)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--device", default="auto")

    args = p.parse_args()

    question = np.load(args.question, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r").reshape(-1)

    if question.ndim != 2 or question.shape[1] != 65:
        raise ValueError(f"question 应该是 (N,65)，现在是 {question.shape}")

    if len(labels) != len(question):
        raise ValueError(f"labels length {len(labels)} != question length {len(question)}")

    n = len(question)
    if n % args.n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={args.n_batches} 整除")

    batch_len = n // args.n_batches
    visible_len = int(round(batch_len * args.visible_ratio))
    pred_len = batch_len - visible_len

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    device = torch.device(device)

    print("[INFO] question:", question.shape)
    print("[INFO] batch_len:", batch_len)
    print("[INFO] visible_len:", visible_len)
    print("[INFO] pred_len:", pred_len)
    print("[INFO] device:", device)

    pred62, meta62 = predict_ml_target(
        question=question,
        labels=labels,
        bundle_path=args.f62_bundle,
        target_index=0,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
    )

    pred63, meta63 = predict_f63_single(
        question=question,
        labels=labels,
        ckpt_path=args.f63_ckpt,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        device=device,
    )

    pred64, meta64 = predict_ml_target(
        question=question,
        labels=labels,
        bundle_path=args.f64_bundle,
        target_index=2,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
    )

    answer = np.zeros((3, args.n_batches, pred_len), dtype=np.float32)
    answer[0] = pred62
    answer[1] = pred63
    answer[2] = pred64

    if not np.isfinite(answer).all():
        raise ValueError("answer contains nan or inf")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, answer.astype(np.float32))

    meta = {
        "question": args.question,
        "labels": args.labels,
        "answer_shape": list(answer.shape),
        "format": "answer[0]=f62 ML, answer[1]=f63 single DL, answer[2]=f64 ML",
        "n_batches": int(args.n_batches),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "pred_len": int(pred_len),
        "f62": meta62,
        "f63": meta63,
        "f64": meta64,
        "ranges": {
            "f62": [float(answer[0].min()), float(answer[0].max()), float(answer[0].mean())],
            "f63": [float(answer[1].min()), float(answer[1].max()), float(answer[1].mean())],
            "f64": [float(answer[2].min()), float(answer[2].max()), float(answer[2].mean())],
        },
    }

    if args.meta_out:
        save_json(meta, args.meta_out)

    print("[OK] saved:", out)
    print("[OK] shape:", answer.shape)
    print("[CHECK] f62 min/max/mean:", meta["ranges"]["f62"])
    print("[CHECK] f63 min/max/mean:", meta["ranges"]["f63"])
    print("[CHECK] f64 min/max/mean:", meta["ranges"]["f64"])


if __name__ == "__main__":
    main()