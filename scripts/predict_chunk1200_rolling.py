#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkdl.config import SELECTED_COLS
from zkdl.dataset import build_input_matrix, time_features
from zkdl.model import DualHeadTCNForecaster
from zkdl.scaler import Standardizer


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


def load_chunk_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt.get("model_kind") != "chunked_joint_tcn":
        raise ValueError(
            f"ckpt model_kind 不是 chunked_joint_tcn: {ckpt.get('model_kind')}"
        )

    cfg = ckpt["config"]

    model = DualHeadTCNForecaster(
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
    target_scaler = Standardizer.from_state_dict(ckpt["target_scaler"])

    return model, input_scaler, target_scaler, cfg, ckpt.get("best", {})


def scale_hist_future(hist_all, fut_all, input_scaler):
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

    return hist_all, fut_all


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--question", required=True, help="data\\official\\question_2d.npy")
    p.add_argument("--labels", required=True, help="runs\\official_modes\\mode_labels.npy")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="输出 pred_f63_f64.npy, shape=(2,10,12000)")
    p.add_argument("--meta-out", default=None)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--chunk-len", type=int, default=None)

    p.add_argument("--device", default="auto")

    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device = torch.device(device)

    model, input_scaler, target_scaler, cfg, best = load_chunk_model(args.ckpt, device)

    question = np.load(args.question)
    labels = np.load(args.labels).reshape(-1)

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

    history_len = int(cfg["history_len"])
    chunk_len = int(args.chunk_len if args.chunk_len is not None else cfg["chunk_len"])
    near_switch_win = int(cfg.get("near_switch_win", 512))

    if history_len > visible_len:
        raise ValueError(
            f"history_len={history_len} > visible_len={visible_len}"
        )

    work = np.asarray(question, dtype=np.float32).copy()
    pred_all = np.zeros((2, args.n_batches, pred_len), dtype=np.float32)

    print("[INFO] question:", question.shape)
    print("[INFO] batch_len:", batch_len)
    print("[INFO] visible_len:", visible_len)
    print("[INFO] pred_len:", pred_len)
    print("[INFO] history_len:", history_len)
    print("[INFO] chunk_len:", chunk_len)
    print("[INFO] device:", device)

    with torch.no_grad():
        for bi in range(args.n_batches):
            bs = bi * batch_len
            be = bs + batch_len

            batch = work[bs:be].copy()
            labels_batch = labels[bs:be]

            offset = 0
            cut = visible_len

            while cut < batch_len:
                clen = min(chunk_len, batch_len - cut)

                hist_all, fut_all, _ = build_input_matrix(
                    batch,
                    labels_batch,
                    selected_cols=SELECTED_COLS,
                    zero_future_target_mask=None,
                    near_switch_win=near_switch_win,
                )

                hist_all, fut_all = scale_hist_future(
                    hist_all=hist_all,
                    fut_all=fut_all,
                    input_scaler=input_scaler,
                )

                hs = cut - history_len
                he = cut
                fs = cut
                fe = cut + clen

                if hs < 0:
                    raise ValueError(
                        f"batch {bi}: hs={hs} < 0, history_len 不够"
                    )

                hist = hist_all[hs:he].astype(np.float32)
                fut = np.concatenate(
                    [fut_all[fs:fe], time_features(clen)],
                    axis=1,
                ).astype(np.float32)

                hist_t = torch.from_numpy(hist[None]).to(device)
                fut_t = torch.from_numpy(fut[None]).to(device)

                raw = model(hist_t, fut_t)[0].cpu().numpy().astype(np.float32)
                pred = target_scaler.inverse_np(raw).astype(np.float32)

                if pred.shape != (clen, 2):
                    raise ValueError(
                        f"batch {bi}, cut {cut}: pred shape {pred.shape} != {(clen, 2)}"
                    )

                batch[fs:fe, 63] = pred[:, 0]
                batch[fs:fe, 64] = pred[:, 1]

                pred_all[0, bi, offset : offset + clen] = pred[:, 0]
                pred_all[1, bi, offset : offset + clen] = pred[:, 1]

                print(
                    f"[ROLL] batch={bi}, cut={cut}, len={clen}, "
                    f"f63_mean={float(pred[:,0].mean()):.6f}, "
                    f"f64_mean={float(pred[:,1].mean()):.6f}",
                    flush=True,
                )

                cut += clen
                offset += clen

            work[bs:be] = batch

    if not np.isfinite(pred_all).all():
        raise ValueError("pred_all contains nan or inf")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred_all.astype(np.float32))

    meta = {
        "question": args.question,
        "labels": args.labels,
        "ckpt": args.ckpt,
        "out": str(out),
        "shape": list(pred_all.shape),
        "format": "pred[0]=f63, pred[1]=f64",
        "n_batches": int(args.n_batches),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "pred_len": int(pred_len),
        "history_len": int(history_len),
        "chunk_len": int(chunk_len),
        "best": best,
        "ranges": {
            "f63": [
                float(pred_all[0].min()),
                float(pred_all[0].max()),
                float(pred_all[0].mean()),
            ],
            "f64": [
                float(pred_all[1].min()),
                float(pred_all[1].max()),
                float(pred_all[1].mean()),
            ],
        },
    }

    if args.meta_out:
        save_json(meta, args.meta_out)

    print("[OK] saved:", out)
    print("[OK] shape:", pred_all.shape)
    print("[CHECK] f63 min/max/mean:", meta["ranges"]["f63"])
    print("[CHECK] f64 min/max/mean:", meta["ranges"]["f64"])


if __name__ == "__main__":
    main()