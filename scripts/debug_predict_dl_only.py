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

from zkdl.dataset import OfficialBatchWindowDataset
from zkdl.model import DualHeadTCNForecaster
from zkdl.scaler import Standardizer


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


def load_model_from_ckpt(ckpt_path, device):
    ckpt_path = str(ckpt_path).strip()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model_kind = cfg.get("model_kind", "direct")
    if model_kind != "direct":
        raise ValueError(f"当前 debug 脚本只支持 direct，model_kind={model_kind}")

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

    return model, input_scaler, target_scaler, cfg


def compute_metrics(pred, true):
    """
    pred/true:
      shape = (2, n_batches, pred_len)
      pred[0]=f63, pred[1]=f64
    """
    if pred.shape != true.shape:
        raise ValueError(f"pred shape {pred.shape} != true shape {true.shape}")

    out = {}
    batch_rows = []

    names = ["f63", "f64"]

    for ti, name in enumerate(names):
        err = pred[ti].astype(np.float64) - true[ti].astype(np.float64)
        out[f"mse_{name}"] = float(np.mean(err * err))
        out[f"mae_{name}"] = float(np.mean(np.abs(err)))
        out[f"bias_{name}"] = float(np.mean(err))

    out["mse_sum"] = float(out["mse_f63"] + out["mse_f64"])

    for bi in range(pred.shape[1]):
        row = {"batch": int(bi), "length": int(pred.shape[2])}

        for ti, name in enumerate(names):
            err = pred[ti, bi].astype(np.float64) - true[ti, bi].astype(np.float64)
            row[f"mse_{name}"] = float(np.mean(err * err))
            row[f"mae_{name}"] = float(np.mean(np.abs(err)))
            row[f"bias_{name}"] = float(np.mean(err))

        row["mse_sum"] = float(row["mse_f63"] + row["mse_f64"])
        batch_rows.append(row)

    return out, batch_rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--question", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--answer", default=None)
    p.add_argument("--metrics-out", default=None)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)
    p.add_argument("--device", default="auto")

    args = p.parse_args()

    question = np.load(args.question)
    labels = np.load(args.labels).reshape(-1)

    if question.ndim != 2 or question.shape[1] != 65:
        raise ValueError(f"question should be (N,65), got {question.shape}")

    if len(labels) != len(question):
        raise ValueError(f"labels length {len(labels)} != question length {len(question)}")

    n = len(question)
    if n % args.n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={args.n_batches} 整除")

    batch_len = n // args.n_batches
    visible_len = int(round(batch_len * args.visible_ratio))
    pred_len = batch_len - visible_len

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    device = torch.device(device)

    model, input_scaler, target_scaler, cfg = load_model_from_ckpt(args.ckpt, device)

    history_len = int(cfg["history_len"])
    near_switch_win = int(cfg.get("near_switch_win", 512))

    print(
        f"[INFO] question={question.shape}, batch_len={batch_len}, "
        f"visible_len={visible_len}, pred_len={pred_len}",
        flush=True,
    )
    print(
        f"[CKPT] history_len={history_len}, hidden={cfg.get('hidden')}, "
        f"levels={cfg.get('levels')}, hist_dim={cfg.get('hist_dim')}, "
        f"future_dim={cfg.get('future_dim')}",
        flush=True,
    )

    if visible_len < history_len:
        raise ValueError(
            f"visible_len={visible_len} < ckpt history_len={history_len}"
        )

    ds = OfficialBatchWindowDataset(
        data=question,
        labels=labels,
        history_len=history_len,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        batch_len=batch_len,
        visible_ratio=args.visible_ratio,
        near_switch_win=near_switch_win,
    )

    pred = np.zeros((2, args.n_batches, pred_len), dtype=np.float32)

    with torch.no_grad():
        for bi in range(len(ds)):
            sample = ds[bi]

            raw = model(
                sample["hist"][None].to(device),
                sample["future"][None].to(device),
            )[0].cpu().numpy()

            pred_np = target_scaler.inverse_np(raw).astype(np.float32)

            pred[0, bi] = pred_np[:, 0]
            pred[1, bi] = pred_np[:, 1]

            print(f"[PRED] batch={bi}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred)

    print("[OK] saved:", out, pred.shape)

    if args.answer:
        ans = np.load(args.answer)

        if ans.ndim != 3 or ans.shape[0] != 3:
            raise ValueError(f"answer should be (3,n_batches,pred_len), got {ans.shape}")

        true = ans[1:3]

        summary, batch_rows = compute_metrics(pred, true)

        result = {
            "question": args.question,
            "labels": args.labels,
            "ckpt": args.ckpt,
            "pred": str(out),
            "answer": args.answer,
            "pred_shape": list(pred.shape),
            "answer_shape": list(ans.shape),
            "summary": summary,
            "batch_metrics": batch_rows,
        }

        metrics_out = Path(args.metrics_out) if args.metrics_out else out.with_suffix(".metrics.json")
        save_json(result, metrics_out)

        print("[METRICS]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("[OK] saved metrics:", metrics_out)


if __name__ == "__main__":
    main()