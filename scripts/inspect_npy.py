#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    args = p.parse_args()

    x = np.load(args.path, mmap_mode="r")

    print("path:", args.path)
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("ndim:", x.ndim)

    if x.ndim == 3:
        print()
        print("按平台描述：")
        print("num_features =", x.shape[0])
        print("num_samples  =", x.shape[1])
        print("seq_len      =", x.shape[2])

        if x.shape[0] == 65:
            print()
            print("判断：这是平台格式 (features, samples, seq_len)")
            print("单个 sample 转成我们模型需要的格式：")
            print("sample_i = x[:, i, :].T")
            print("sample_i.shape = (seq_len, 65)")

    if x.ndim == 2:
        print()
        print("判断：这是旧格式/内部格式 (sequence_length, features)")
        print("shape = (N, 65)")


if __name__ == "__main__":
    main()