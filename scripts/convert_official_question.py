#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="官方题目集.npy, shape=(65,10,60000)")
    p.add_argument("--out", required=True, help="转换后的 question.npy, shape=(600000,65)")
    args = p.parse_args()

    x = np.load(args.input, mmap_mode="r")

    if x.ndim != 3:
        raise ValueError(f"expected 3D official format, got {x.shape}")

    if x.shape[0] != 65:
        raise ValueError(f"expected first dim num_features=65, got {x.shape}")

    num_features, num_samples, seq_len = x.shape

    # (65, 10, 60000) -> (10, 60000, 65) -> (600000, 65)
    q = np.asarray(x.transpose(1, 2, 0).reshape(num_samples * seq_len, num_features), dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, q)

    print("[OK] official input:", x.shape)
    print("[OK] converted question:", q.shape)
    print("[OK] saved:", out)

    visible_len = int(round(seq_len * 0.8))
    pred_len = seq_len - visible_len

    tail = q.reshape(num_samples, seq_len, num_features)[:, visible_len:, [62, 63, 64]]
    print("[INFO] num_samples:", num_samples)
    print("[INFO] seq_len:", seq_len)
    print("[INFO] visible_len:", visible_len)
    print("[INFO] pred_len:", pred_len)
    print("[CHECK] target tail abs mean:", float(np.mean(np.abs(tail))))
    print("[CHECK] target tail max abs:", float(np.max(np.abs(tail))))


if __name__ == "__main__":
    main()