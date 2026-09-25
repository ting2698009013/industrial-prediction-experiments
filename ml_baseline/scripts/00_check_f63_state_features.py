import argparse, sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from zkmm.data import load_train
from zkmm.features_direct import build_direct_features

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--labels", default=None)
    args = p.parse_args()
    data = load_train(args.data, mmap=True)
    labels = np.load(args.labels) if args.labels else None
    cuts = np.array([48000, 108000, 300000], dtype=np.int64)
    horizons = np.array([0, 1000, 5000], dtype=np.int64)
    xb, _, nb = build_direct_features(data, 1, cuts, horizons, labels=labels, cross="none", f63_state="base")
    xs, _, ns = build_direct_features(data, 1, cuts, horizons, labels=labels, cross="none", f63_state="strong")
    print("base shape:", xb.shape, "n_names:", len(nb))
    print("strong shape:", xs.shape, "n_names:", len(ns))
    print("extra columns:", xs.shape[1] - xb.shape[1])
    print("strong names tail:", ns[-40:])

if __name__ == "__main__":
    main()
