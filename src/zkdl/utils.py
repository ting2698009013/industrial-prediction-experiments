from __future__ import annotations
from pathlib import Path
import json, random
import numpy as np
import torch
from .config import KNOWN_SEGMENTS, BATCH_LEN, VISIBLE_RATIO, N_MODES, DL_TARGET_NAMES

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default), encoding="utf-8")

def labels_from_segments(n: int, segments=KNOWN_SEGMENTS):
    labels = np.full(n, -1, dtype=np.int16)
    for s in segments:
        labels[int(s["start"]):int(s["end"])] = int(s["label"])
    if np.any(labels < 0):
        last = 0
        for i in range(n):
            if labels[i] >= 0: last = labels[i]
            else: labels[i] = last
    return labels

def fold_bounds(fold: int, batch_len=BATCH_LEN, visible_ratio=VISIBLE_RATIO):
    visible_len = int(round(batch_len * visible_ratio))
    start = fold * batch_len
    cut = start + visible_len
    end = start + batch_len
    return start, cut, end

def target_mse_np(pred, true, names=None):
    p, t = np.asarray(pred), np.asarray(true)
    if names is None: names = DL_TARGET_NAMES
    if p.ndim == 2 and p.shape[0] == len(names):
        p, t = p.T, t.T
    out = {}
    for i, name in enumerate(names):
        e = p[:,i].astype(np.float64) - t[:,i].astype(np.float64)
        out[f"mse_{name}"] = float(np.mean(e*e))
        out[f"mae_{name}"] = float(np.mean(np.abs(e)))
        out[f"bias_{name}"] = float(np.mean(e))
    out["mse_sum"] = float(sum(out[f"mse_{name}"] for name in names))
    return out

def mode_onehot(labels, n_modes=N_MODES):
    labels = np.asarray(labels).astype(np.int64)
    out = np.zeros((len(labels), n_modes), dtype=np.float32)
    ok = (labels >= 0) & (labels < n_modes)
    out[np.arange(len(labels))[ok], labels[ok]] = 1.0
    return out

def mode_switch_features(labels, near_win=512, n_modes=N_MODES):
    """返回 mode one-hot + time_since_mode_start_norm + is_near_switch，共 n_modes+2 维。"""
    if labels is None:
        return None
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)
    oh = mode_onehot(labels, n_modes=n_modes)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    seg_start = np.zeros(n, dtype=np.int64)
    last = 0
    for ch in list(changes) + [n]:
        seg_start[last:ch] = last
        last = ch
    since = (np.arange(n, dtype=np.float32) - seg_start.astype(np.float32))
    since = np.minimum(since / max(float(near_win), 1.0), 1.0).reshape(-1, 1)
    near = np.zeros(n, dtype=np.float32)
    for ch in changes:
        s = max(0, int(ch) - int(near_win))
        e = min(n, int(ch) + int(near_win) + 1)
        near[s:e] = 1.0
    return np.concatenate([oh, since.astype(np.float32), near.reshape(-1,1)], axis=1).astype(np.float32)

def time_features(length):
    pos = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([pos, np.sin(2*np.pi*pos), np.cos(2*np.pi*pos)], axis=1).astype(np.float32)
