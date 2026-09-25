from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zkml.data import load_clean_data, make_question_from_full


def main():
    ap = argparse.ArgumentParser(description="从完整训练集模拟平台 question/answer。")
    ap.add_argument("--data", required=True, help="full data, shape=(600000,65)")
    ap.add_argument("--question-out", required=True)
    ap.add_argument("--answer-out", default=None)
    args = ap.parse_args()
    full = load_clean_data(args.data)
    q, ans = make_question_from_full(full)
    np.save(args.question_out, q.astype(np.float32))
    print(f"[SAVE] question {args.question_out}, shape={q.shape}, dtype={q.dtype}", flush=True)
    if args.answer_out:
        np.save(args.answer_out, ans.astype(np.float32))
        print(f"[SAVE] answer {args.answer_out}, shape={ans.shape}, dtype={ans.dtype}", flush=True)

if __name__ == "__main__":
    main()
