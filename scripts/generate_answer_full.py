#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkmm.config import SELECTED_COLS
from zkmm.mode import mode_onehot, time_since_segment_start, is_near_switch

from zkdl.dataset import OfficialBatchWindowDataset
from zkdl.model import DualHeadTCNForecaster
from zkdl.scaler import Standardizer


TARGET_COLS = [62, 63, 64]


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


def _safe_div(a, b):
    return a / (b + 1e-8)


def _lag(y, cut, lags=(1, 2, 3, 5, 10, 20, 50, 100, 500, 1000)):
    return [float(y[max(0, cut - l)]) for l in lags]


def _stats(y, cut, windows=(5, 10, 20, 50, 100, 500, 1000)):
    out = []

    for w in windows:
        s = max(0, cut - w)
        seg = y[s:cut].astype("float64")

        if len(seg) == 0:
            out += [0.0, 0.0, 0.0, 0.0]
        else:
            out += [
                float(seg.mean()),
                float(seg.std()),
                float(seg.min()),
                float(seg.max()),
            ]

    for w in (100, 500, 1000):
        s = max(0, cut - w)
        seg = y[s:cut].astype("float64")
        if len(seg) >= 2:
            out.append(float((seg[-1] - seg[0]) / max(1, len(seg) - 1)))
        else:
            out.append(0.0)

    return out


def _get_f62_target_bundle(bundle):
    if "target_0" in bundle:
        return bundle["target_0"]

    if "targets" in bundle and "target_0" in bundle["targets"]:
        return bundle["targets"]["target_0"]

    raise ValueError("f62 bundle 中找不到 target_0")


def build_f62_features_for_batch(
    batch,
    labels_batch=None,
    selected_cols=SELECTED_COLS,
    visible_ratio=0.8,
    horizon_scale=12000,
    add_mode=True,
    add_switch=True,
):
    """
    动态版本 f62 特征构造。

    batch:
      shape = (batch_len, 65)

    输出:
      X:
        shape = (pred_len, n_features)

    和旧 ML 逻辑保持一致：
      target_index=0 时训练 residual = f62 - f01
      预测阶段用 pred_f62 = f01_future + residual
    """
    if batch.ndim != 2 or batch.shape[1] != 65:
        raise ValueError(f"batch should be (batch_len,65), got {batch.shape}")

    batch_len = len(batch)
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    cuts = np.full(pred_len, visible_len, dtype=np.int64)
    horizons = np.arange(pred_len, dtype=np.int64)
    t_idx = cuts + horizons

    Xraw = batch[:, selected_cols].astype(np.float32, copy=False)

    parts = []
    names = []

    # 当前时刻 process 特征，官方题目里 f0~f61 future 已知
    parts.append(Xraw[t_idx])
    names += [f"f{int(c):02d}_t" for c in selected_cols]

    # process lag1
    lag = np.maximum(t_idx - 1, 0)
    parts.append(Xraw[lag])
    names += [f"f{int(c):02d}_lag1" for c in selected_cols]

    # process diff1
    parts.append(Xraw[t_idx] - Xraw[lag])
    names += [f"f{int(c):02d}_diff1" for c in selected_cols]

    # horizon 特征
    h = horizons.astype(np.float32) + 1.0
    horizon_scale = float(horizon_scale)
    parts.append(
        np.column_stack(
            [
                h / horizon_scale,
                np.log1p(h) / np.log1p(horizon_scale),
                np.sqrt(h) / np.sqrt(horizon_scale),
            ]
        ).astype(np.float32)
    )
    names += ["h_norm", "h_log", "h_sqrt"]

    # f62 residual 历史状态：只用 visible_len 之前
    state_y = batch[:, 62].astype(np.float32) - batch[:, 1].astype(np.float32)

    rows = []
    for c in cuts:
        c = int(c)
        row = (
            _lag(state_y, c)
            + _stats(state_y, c)
            + [
                float(state_y[c - 1]),
                float(np.mean(state_y[max(0, c - 3000):c])),
            ]
        )
        rows.append(row)

    parts.append(np.asarray(rows, dtype=np.float32))
    names += [f"y_state_{i}" for i in range(len(rows[0]))]

    # f62 最关键 anchor：future f01
    parts.append(batch[t_idx, 1].astype(np.float32)[:, None])
    names.append("f01_future_anchor")

    # mode 特征
    if labels_batch is not None and add_mode:
        lab = labels_batch[t_idx].astype(int)
        parts.append(mode_onehot(lab))
        names += ["mode_0", "mode_1", "mode_2"]

        if add_switch:
            ts = time_since_segment_start(labels_batch)[t_idx] / 60000.0
            ns = is_near_switch(labels_batch)[t_idx]
            parts.append(np.column_stack([ts, ns]).astype(np.float32))
            names += ["time_since_mode_start_norm", "is_near_switch"]

    X = np.hstack(parts).astype(np.float32)

    return X, names


def predict_f62_all_batches(
    question,
    labels,
    f62_bundle_path,
    n_batches=10,
    visible_ratio=0.8,
    horizon_scale=12000,
):
    bundle = joblib.load(f62_bundle_path)
    tb = _get_f62_target_bundle(bundle)

    model = tb["global_model"]

    n = len(question)
    if n % n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    pred = np.zeros((n_batches, pred_len), dtype=np.float32)

    for bi in range(n_batches):
        bs = bi * batch_len
        be = bs + batch_len
        cut = bs + visible_len

        batch = np.array(question[bs:be], dtype=np.float32, copy=True)
        labels_batch = labels[bs:be] if labels is not None else None

        X, names = build_f62_features_for_batch(
            batch=batch,
            labels_batch=labels_batch,
            visible_ratio=visible_ratio,
            horizon_scale=horizon_scale,
            add_mode=True,
            add_switch=tb.get("add_switch", True),
        )

        residual = model.predict(X).astype(np.float32)

        if "f62_res_q001" in tb and "f62_res_q999" in tb:
            residual = np.clip(
                residual,
                float(tb["f62_res_q001"]),
                float(tb["f62_res_q999"]),
            )

        f01_future = batch[visible_len:be - bs, 1].astype(np.float32)
        pred[bi] = f01_future + residual

        print(f"[F62] batch={bi}, pred_len={pred_len}", flush=True)

    return pred


def load_dl_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        raise ValueError("DL ckpt 里没有 config")

    cfg = ckpt["config"]
    model_kind = cfg.get("model_kind", "direct")

    if model_kind != "direct":
        raise ValueError(f"当前只支持 direct DL 模型，ckpt model_kind={model_kind}")

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


def predict_f63_f64_all_batches(
    question,
    labels,
    dl_ckpt,
    n_batches=10,
    visible_ratio=0.8,
    device="auto",
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    model, input_scaler, target_scaler, cfg = load_dl_model(dl_ckpt, device)

    n = len(question)
    if n % n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={n_batches} 整除")

    batch_len = n // n_batches
    visible_len = int(round(batch_len * visible_ratio))
    pred_len = batch_len - visible_len

    history_len = int(cfg["history_len"])
    if visible_len < history_len:
        raise ValueError(
            f"当前 question visible_len={visible_len} < DL history_len={history_len}。"
            f"请调整 pseudo split，或换 history_len 更短的模型。"
        )

    near_switch_win = int(cfg.get("near_switch_win", 512))

    ds = OfficialBatchWindowDataset(
        data=question,
        labels=labels,
        history_len=history_len,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        batch_len=batch_len,
        visible_ratio=visible_ratio,
        near_switch_win=near_switch_win,
    )

    pred = np.zeros((2, n_batches, pred_len), dtype=np.float32)

    with torch.no_grad():
        for bi in range(len(ds)):
            sample = ds[bi]

            hist = sample["hist"][None].to(device)
            future = sample["future"][None].to(device)

            raw = model(hist, future)[0].cpu().numpy().astype(np.float32)
            pred_np = target_scaler.inverse_np(raw).astype(np.float32)

            pred[0, bi] = pred_np[:, 0]  # f63
            pred[1, bi] = pred_np[:, 1]  # f64

            print(f"[DL] batch={bi}, pred_len={pred_len}", flush=True)

    return pred


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--question", required=True, help="question.npy, shape=(N,65)")
    p.add_argument("--labels", required=True, help="mode_labels.npy, shape=(N,)")

    p.add_argument("--f62-bundle", required=True, help="f62_model_bundle.joblib")
    p.add_argument("--dl-ckpt", required=True, help="DL best.pt for f63/f64")

    p.add_argument("--out", required=True, help="output answer.npy")
    p.add_argument("--meta-out", default=None)

    p.add_argument("--n-batches", type=int, default=10)
    p.add_argument("--visible-ratio", type=float, default=0.8)

    # f62 ML 训练时 horizon 特征按 12000 归一化，官方数据就是 12000。
    # pseudo_valid 较短时仍建议保持 12000，减少和训练分布不一致。
    p.add_argument("--f62-horizon-scale", type=int, default=12000)

    p.add_argument("--device", default="auto")

    args = p.parse_args()

    question = np.load(args.question)
    labels = np.load(args.labels).reshape(-1)

    if question.ndim != 2 or question.shape[1] != 65:
        raise ValueError(f"question 应该是 shape=(N,65)，现在是 {question.shape}")

    if len(labels) != len(question):
        raise ValueError(f"labels length {len(labels)} != question length {len(question)}")

    n = len(question)

    if n % args.n_batches != 0:
        raise ValueError(f"N={n} 不能被 n_batches={args.n_batches} 整除")

    batch_len = n // args.n_batches
    visible_len = int(round(batch_len * args.visible_ratio))
    pred_len = batch_len - visible_len

    print(
        f"[INFO] question={question.shape}, "
        f"batch_len={batch_len}, visible_len={visible_len}, pred_len={pred_len}",
        flush=True,
    )

    pred_f62 = predict_f62_all_batches(
        question=question,
        labels=labels,
        f62_bundle_path=args.f62_bundle,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        horizon_scale=args.f62_horizon_scale,
    )

    pred_f63_f64 = predict_f63_f64_all_batches(
        question=question,
        labels=labels,
        dl_ckpt=args.dl_ckpt,
        n_batches=args.n_batches,
        visible_ratio=args.visible_ratio,
        device=args.device,
    )

    answer = np.zeros((3, args.n_batches, pred_len), dtype=np.float32)

    answer[0] = pred_f62
    answer[1] = pred_f63_f64[0]
    answer[2] = pred_f63_f64[1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, answer.astype(np.float32))

    meta = {
        "question": args.question,
        "labels": args.labels,
        "f62_bundle": args.f62_bundle,
        "dl_ckpt": args.dl_ckpt,
        "out": str(out),
        "answer_shape": list(answer.shape),
        "answer_format": "answer[0]=f62, answer[1]=f63, answer[2]=f64",
        "n_batches": int(args.n_batches),
        "batch_len": int(batch_len),
        "visible_len": int(visible_len),
        "pred_len": int(pred_len),
        "visible_ratio": float(args.visible_ratio),
        "f62_horizon_scale": int(args.f62_horizon_scale),
    }

    meta_out = Path(args.meta_out) if args.meta_out else out.with_suffix(".meta.json")
    save_json(meta, meta_out)

    print("[OK] saved answer:", out, answer.shape)
    print("[OK] saved meta:", meta_out)


if __name__ == "__main__":
    main()