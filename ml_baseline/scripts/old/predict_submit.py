from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkml.config import TARGET_NAMES
from zkml.infer import recursive_predict_batch, question_to_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="platform question npy, shape=(65,10,60000)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--labels-question", default=None, help="optional labels for question, shape=(10,60000) or (600000,)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    q = np.load(args.question).astype(np.float32, copy=False)
    batches = question_to_batches(q)
    model_dir = Path(args.model_dir)
    models = [load(model_dir / f"model_{name}.joblib") for name in TARGET_NAMES]

    labels = None
    if args.labels_question:
        labels = np.load(args.labels_question).astype(np.int16, copy=False)
        if labels.shape == (600000,):
            labels = labels.reshape(10, 60000)
        if labels.shape != (10, 60000):
            raise ValueError(f"labels-question shape should be (10,60000) or (600000,), got {labels.shape}")

    pred = np.zeros((3, 10, 12000), dtype=np.float32)
    for b in range(10):
        print(f"[PRED] batch {b}", flush=True)
        lab_b = labels[b] if labels is not None else None
        pred[:, b, :] = recursive_predict_batch(batches[b], models, labels=lab_b)

    np.save(args.out, pred.astype(np.float32))
    print(f"[SAVE] {args.out}, shape={pred.shape}, dtype={pred.dtype}", flush=True)


if __name__ == "__main__":
    main()
