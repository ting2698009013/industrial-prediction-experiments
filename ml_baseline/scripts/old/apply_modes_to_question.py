"""Apply trained mode classifier to official question.npy, shape=(65,10,60000)."""
from __future__ import annotations
import argparse, json, os
import joblib
import numpy as np


def rolling_mean_2d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32, copy=True)
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    xp = np.pad(x.astype(np.float64), ((pad_left, pad_right), (0, 0)), mode="edge")
    cs = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(xp, axis=0)])
    return ((cs[win:] - cs[:-win]) / float(win)).astype(np.float32)


def build_mode_features(data_2d: np.ndarray, cols: list[int], smooth_win: int) -> np.ndarray:
    raw = np.asarray(data_2d[:, cols], dtype=np.float32)
    sm = rolling_mean_2d(raw, smooth_win)
    diff = np.zeros_like(sm); diff[1:] = sm[1:] - sm[:-1]
    return np.hstack([sm, diff]).astype(np.float32)


def majority_smooth(labels: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return labels.copy()
    scores = []
    for m in range(3):
        scores.append(rolling_mean_2d((labels == m).astype(np.float32)[:, None], win)[:, 0])
    return np.argmax(np.vstack(scores), axis=0).astype(np.int16)


def segments_from_labels(labels):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]; ends = np.r_[change, len(labels)]
    return [(int(s), int(e), int(labels[s]), int(e-s)) for s,e in zip(starts, ends)]


def merge_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    out = labels.astype(np.int16, copy=True)
    changed = True
    while changed:
        changed = False
        segs = segments_from_labels(out)
        if len(segs) <= 1: break
        for k, (s,e,lab,L) in enumerate(segs):
            if L >= min_len: continue
            if k == 0: new = segs[k+1][2]
            elif k == len(segs)-1: new = segs[k-1][2]
            else: new = segs[k-1][2] if segs[k-1][3] >= segs[k+1][3] else segs[k+1][2]
            out[s:e] = new; changed = True; break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="question npy shape=(65,10,60000)")
    ap.add_argument("--model", required=True, help="mode_classifier.joblib")
    ap.add_argument("--out", default="runs/question_modes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    q = np.load(args.question, mmap_mode="r")
    if q.shape[0] != 65:
        raise ValueError(f"expected question shape=(65,10,60000), got {q.shape}")
    obj = joblib.load(args.model)
    model, cols, smooth_win, min_len = obj["model"], obj["cols"], int(obj["smooth_win"]), int(obj["min_segment_len"])

    labels_batches = []
    summaries = []
    for b in range(q.shape[1]):
        data_2d = q[:, b, :].T  # (60000,65)
        X_feat = build_mode_features(data_2d, cols, smooth_win)
        lab = model.predict(X_feat).astype(np.int16)
        lab = majority_smooth(lab, max(101, smooth_win // 2))
        lab = merge_short_segments(lab, min_len=min(min_len, 3000))
        labels_batches.append(lab)
        u, c = np.unique(lab, return_counts=True)
        summaries.append({"batch": b, "counts": {str(int(a)): int(bb) for a, bb in zip(u, c)}, "segments": segments_from_labels(lab)})

    labels_batches = np.stack(labels_batches, axis=0)  # (10,60000)
    np.save(os.path.join(args.out, "question_mode_labels.npy"), labels_batches.astype(np.int16))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"question_shape": list(q.shape), "cols": cols, "batches": summaries}, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved {args.out}/question_mode_labels.npy shape={labels_batches.shape}")

if __name__ == "__main__":
    main()
