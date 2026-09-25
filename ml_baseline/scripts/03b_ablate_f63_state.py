import argparse, subprocess, sys, json
from pathlib import Path
import pandas as pd

def run(cmd):
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", default="2,5,9", help="建议先用 2,5,9；确认后再 all")
    ap.add_argument("--out", default="runs/f63_ablation")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--row-stride", default="10")
    ap.add_argument("--cut-stride", default="3000")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("onehot_base", "01_eval_mode_onehot_direct.py", ["--cross", "none", "--f63-state", "base"]),
        ("modespec_base_blend0", "02_eval_mode_specific_direct.py", ["--cross", "none", "--f63-state", "base", "--blend-global-weight", "0"]),
        ("modespec_base_blend015", "02_eval_mode_specific_direct.py", ["--cross", "none", "--f63-state", "base", "--blend-global-weight", "0.15"]),
        ("modespec_strong_blend0", "02_eval_mode_specific_direct.py", ["--cross", "none", "--f63-state", "strong", "--blend-global-weight", "0"]),
        ("modespec_strong_blend015", "02_eval_mode_specific_direct.py", ["--cross", "none", "--f63-state", "strong", "--blend-global-weight", "0.15"]),
    ]

    for name, script, extra in experiments:
        cmd = [
            sys.executable, str(Path("scripts") / script),
            "--data", args.data,
            "--labels", args.labels,
            "--folds", args.folds,
            "--model", args.model,
            "--row-stride", str(args.row_stride),
            "--cut-stride", str(args.cut_stride),
            "--out", str(out / name),
        ] + extra
        if args.fast:
            cmd.append("--fast")
        run(cmd)

    # 汇总各实验 summary
    rows = []
    for name, _, _ in experiments:
        sp = out / name / "summary.json"
        if sp.exists():
            obj = json.loads(sp.read_text(encoding="utf-8"))
            rows.append({
                "experiment": name,
                "mean_mse_sum": obj.get("mean_mse_sum"),
                "mean_mse_f62": obj.get("mean_mse_f62"),
                "mean_mse_f63": obj.get("mean_mse_f63"),
                "mean_mse_f64": obj.get("mean_mse_f64"),
            })
    if rows:
        df = pd.DataFrame(rows).sort_values("mean_mse_sum")
        df.to_csv(out / "comparison.csv", index=False)
        print("\n[SUMMARY]")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
