"""
传统ML极致优化 — 分模式建模 + 多尺度 + Stacking集成
=====================================================
快速版: 只用快速模型 (XGB, LGBM, CatBoost, Ridge)
"""

import os, sys, time, warnings
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_COLS = np.array([0, 1, 7, 8, 9, 10, 12, 16, 18, 19, 20, 21, 22,
                          25, 26, 27, 28, 29, 30, 31, 37, 47, 48, 50, 51,
                          52, 53, 54, 55, 57, 59], dtype=np.int32)

def _fi(v): return int(np.where(SELECTED_COLS == v)[0][0])
F01=_fi(1); F22=_fi(22); F48=_fi(48); F50=_fi(50); F51=_fi(51)
F52=_fi(52); F53=_fi(53); F54=_fi(54)

BASELINE = {"f62": 0.000152, "f63": 0.185557, "f64": 0.172042}


def load_data():
    clean = np.load(os.path.join(BASE_DIR, "output", "03_outliers", "data_after_outlier_handling.npy")).astype(np.float64)
    labels = np.load(os.path.join(BASE_DIR, "output", "02_modes", "mode_labels.npy"))
    X = clean[:, SELECTED_COLS].copy()
    y = clean[:, [62, 63, 64]].copy()
    print(f"  数据加载完成: X={X.shape}, 模式分布={np.bincount(labels.astype(int))}", flush=True)
    return X, y, labels


def get_folds(n=600000, n_splits=10, val_ratio=0.2):
    seg = n // n_splits; vs = int(seg * val_ratio)
    folds = []
    for f in range(n_splits):
        seg_i = n_splits - 1 - f
        s = seg_i * seg
        folds.append((np.arange(0, s + seg - vs), np.arange(s + seg - vs, s + seg)))
    return folds


def build_ts(X, y_col, windows=(5,10,20), mode_labels=None,
             cross_pairs=None, cross_ops=("mul","sub","div"),
             y_lags=20, add_ewm=True):
    N, F = X.shape
    w_max = max(max(windows), y_lags)
    t_idx = np.arange(w_max, N)
    fnames = [f"f{SELECTED_COLS[i]:02d}" for i in range(F)]
    parts, names = [], []

    # 每个窗口: rmean + diff + y_rmean
    for w in windows:
        cs = np.vstack([np.zeros((1,F)), np.cumsum(X, axis=0)])
        parts.append((cs[t_idx] - cs[t_idx-w]) / w)
        names += [f"{n}_rm{w}" for n in fnames]
        parts.append(X[w_max:] - X[w_max-w:N-w])
        names += [f"{n}_df{w}" for n in fnames]
        csy = np.concatenate([[0], np.cumsum(y_col)])
        parts.append(((csy[t_idx]-csy[t_idx-w])/w)[:, np.newaxis])
        names.append(f"y_rm{w}")

    # raw + lag1
    parts.append(X[w_max:]); names += [f"{n}_raw" for n in fnames]
    parts.append(X[w_max-1:N-1]); names += [f"{n}_lag1" for n in fnames]

    # y_lags
    for lag in range(1, y_lags + 1):
        parts.append(y_col[w_max-lag:N-lag][:, np.newaxis]); names.append(f"y_lag{lag}")

    # EWM
    if add_ewm:
        for span in [5, 10, 20]:
            alpha = 2.0 / (span + 1)
            ewm = np.empty(N); ewm[0] = y_col[0]
            for i in range(1, N): ewm[i] = alpha * y_col[i] + (1 - alpha) * ewm[i-1]
            parts.append(ewm[w_max:N][:, np.newaxis]); names.append(f"y_ewm{span}")

    # 交叉 × - ÷
    if cross_pairs:
        orig = X[w_max:]
        for (i, j) in cross_pairs:
            ni, nj = fnames[i], fnames[j]
            for op in cross_ops:
                if op == "mul":
                    parts.append((orig[:, i] * orig[:, j])[:, np.newaxis]); names.append(f"{ni}x{nj}")
                elif op == "sub":
                    parts.append((orig[:, i] - orig[:, j])[:, np.newaxis]); names.append(f"{ni}-{nj}")
                elif op == "div":
                    parts.append((orig[:, i] / (orig[:, j] + 1e-8))[:, np.newaxis]); names.append(f"{ni}d{nj}")

    # mode
    if mode_labels is not None:
        ml = mode_labels[w_max:].astype(np.float32)
        oh = np.zeros((len(ml), 3), dtype=np.float32)
        for m in range(3): oh[:, m] = (ml == m).astype(np.float32)
        parts.append(oh); names += ["mode_0", "mode_1", "mode_2"]

    return np.hstack(parts).astype(np.float32), y_col[w_max:], names


def prepare_val(X_tr, y_tr, X_va, y_va, w_max, mode_tr=None, mode_va=None, **kw):
    """拼接训练尾部给验证集提供上下文"""
    ml_tr = mode_tr if mode_tr is not None else None
    X_t, y_t, _ = build_ts(X_tr, y_tr, mode_labels=ml_tr, **kw)

    ctx = w_max + 5
    X_vc = np.vstack([X_tr[-ctx:], X_va])
    y_vc = np.concatenate([y_tr[-ctx:], y_va])
    ml_vc = np.concatenate([mode_tr[-ctx:], mode_va]) if mode_tr is not None else None

    X_v_full, y_v_full, _ = build_ts(X_vc, y_vc, mode_labels=ml_vc, **kw)
    nv = len(X_va)
    return X_t, y_t, X_v_full[-nv:], y_v_full[-nv:]


def get_fast_models():
    return {
        "xgb": xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.1,
                                 max_leaves=64, min_child_weight=50,
                                 subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=-1, random_state=42, verbosity=0),
        "lgbm": lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.1,
                                   num_leaves=64, min_child_samples=50,
                                   subsample=0.8, colsample_bytree=0.8,
                                   n_jobs=-1, random_state=42, verbose=-1),
        "catboost": CatBoostRegressor(iterations=500, depth=8, learning_rate=0.1,
                                      l2_leaf_reg=10, random_seed=42, verbose=0),
        "ridge": Ridge(alpha=1.0),
    }


def stacking_fit_predict(X_tr, y_tr, X_va, verbose=True):
    """Holdout-based Stacking: 返回验证集预测"""
    n = len(X_tr)
    hold_size = min(50000, n // 5)
    X_train, y_train = X_tr[:-hold_size], y_tr[:-hold_size]
    X_hold, y_hold = X_tr[-hold_size:], y_tr[-hold_size:]

    models = get_fast_models()
    hold_preds = np.zeros((hold_size, len(models)))
    trained = {}

    for mi, (name, model) in enumerate(models.items()):
        t0 = time.time()
        m = clone(model)
        m.fit(X_train, y_train)
        hold_preds[:, mi] = m.predict(X_hold)
        trained[name] = m
        if verbose:
            print(f"          {name}: {time.time()-t0:.1f}s", flush=True)

    # Level-1: Ridge on holdout
    meta = Ridge(alpha=1.0)
    meta.fit(hold_preds, y_hold)

    # 预测验证集
    va_preds = np.column_stack([trained[n].predict(X_va) for n in models])
    return meta.predict(va_preds), trained, meta


def normalize(X_tr, y_tr, X_va, y_va):
    xm, xs = X_tr.mean(0), X_tr.std(0); xs[xs==0]=1
    ym, ys = y_tr.mean(), max(y_tr.std(), 1e-8)
    return ((X_tr-xm)/xs).astype(np.float32), ((y_tr-ym)/ys).astype(np.float32), \
           ((X_va-xm)/xs).astype(np.float32), ((y_va-ym)/ys).astype(np.float32), ym, ys


# ============================================================
# 实验1: 全局 Stacking (多尺度+扩展交叉)
# ============================================================
def run_global_stacking(X_raw, y_col, labels, folds, target, cross_pairs,
                        residual_col=None, **kw):
    print(f"\n  === {target}: 全局Stacking ===", flush=True)
    rows = []
    for fi, (tri, vai) in enumerate(folds):
        t0 = time.time()
        X_tr, y_tr = X_raw[tri], y_col[tri]
        X_va, y_va = X_raw[vai], y_col[vai]

        # 残差建模
        if residual_col is not None:
            y_tr_r = y_tr - X_tr[:, residual_col]
            y_va_r = y_va - X_va[:, residual_col]
        else:
            y_tr_r, y_va_r = y_tr.copy(), y_va.copy()

        Xn_tr, yn_tr, Xn_va, yn_va, ym, ys = normalize(X_tr, y_tr_r, X_va, y_va_r)
        Xt, yt, Xv, yv = prepare_val(Xn_tr, yn_tr, Xn_va, yn_va,
                                       max(kw.get("windows",(5,10,20))),
                                       mode_tr=labels[tri], mode_va=labels[vai], **kw)

        pred, _, _ = stacking_fit_predict(Xt, yt, Xv, verbose=(fi<=1))
        y_pred = pred * ys + ym
        if residual_col is not None:
            y_pred += X_va[:, residual_col]
        mse = float(np.mean((y_va - y_pred)**2))
        elapsed = time.time() - t0
        rows.append({"fold": fi, "mse": mse, "time": round(elapsed,1), "n_feat": Xt.shape[1]})
        print(f"    F{fi}: MSE={mse:.8f} ({elapsed:.0f}s) nf={Xt.shape[1]}", flush=True)

    mse_avg = np.mean([r["mse"] for r in rows])
    print(f"    >> 平均 MSE={mse_avg:.8f}", flush=True)
    return mse_avg


# ============================================================
# 实验2: 分模式 Stacking
# ============================================================
def run_mode_stacking(X_raw, y_col, labels, folds, target, cross_pairs, **kw):
    print(f"\n  === {target}: 分模式Stacking ===", flush=True)
    rows = []
    for fi, (tri, vai) in enumerate(folds):
        t0 = time.time()
        fold_pred = np.zeros(len(vai))

        for mode in range(3):
            va_idx = np.where(labels[vai] == mode)[0]
            if len(va_idx) == 0: continue

            tr_idx = np.where(labels[tri] == mode)[0]
            if len(tr_idx) < 1000: tr_idx = tri  # 回退全局

            X_tr_m, y_tr_m = X_raw[tr_idx], y_col[tr_idx]
            X_va_m, y_va_m = X_raw[vai[va_idx]], y_col[vai[va_idx]]

            Xn_tr, yn_tr, Xn_va, yn_va, ym, ys = normalize(X_tr_m, y_tr_m, X_va_m, y_va_m)
            try:
                Xt, yt, Xv, yv = prepare_val(Xn_tr, yn_tr, Xn_va, yn_va,
                                               max(kw.get("windows",(5,10,20))), **kw)
            except Exception as e:
                print(f"      [W] mode={mode} err: {e}", flush=True)
                continue

            if len(Xt) < 100: continue
            pred, _, _ = stacking_fit_predict(Xt, yt, Xv, verbose=False)
            fold_pred[va_idx] = pred * ys + ym

        mse = float(np.mean((y_col[vai] - fold_pred)**2))
        elapsed = time.time() - t0
        rows.append({"fold": fi, "mse": mse, "time": round(elapsed,1)})
        print(f"    F{fi}: MSE={mse:.8f} ({elapsed:.0f}s)", flush=True)

    mse_avg = np.mean([r["mse"] for r in rows])
    print(f"    >> 平均 MSE={mse_avg:.8f}", flush=True)
    return mse_avg


# ============================================================
# 实验3: 多模型平均 (不Stacking, 直接平均)
# ============================================================
def run_model_avg(X_raw, y_col, labels, folds, target, cross_pairs, **kw):
    print(f"\n  === {target}: 多模型直接平均 ===", flush=True)
    rows = []
    for fi, (tri, vai) in enumerate(folds):
        t0 = time.time()
        X_tr, y_tr = X_raw[tri], y_col[tri]
        X_va, y_va = X_raw[vai], y_col[vai]

        Xn_tr, yn_tr, Xn_va, yn_va, ym, ys = normalize(X_tr, y_tr, X_va, y_va)
        Xt, yt, Xv, yv = prepare_val(Xn_tr, yn_tr, Xn_va, yn_va,
                                       max(kw.get("windows",(5,10,20))),
                                       mode_tr=labels[tri], mode_va=labels[vai], **kw)

        models = get_fast_models()
        preds = []
        for name, model in models.items():
            m = clone(model)
            m.fit(Xt, yt)
            preds.append(m.predict(Xv) * ys + ym)

        avg_pred = np.mean(preds, axis=0)
        mse = float(np.mean((y_va - avg_pred)**2))
        elapsed = time.time() - t0
        rows.append({"fold": fi, "mse": mse, "time": round(elapsed,1)})
        print(f"    F{fi}: MSE={mse:.8f} ({elapsed:.0f}s)", flush=True)

    mse_avg = np.mean([r["mse"] for r in rows])
    print(f"    >> 平均 MSE={mse_avg:.8f}", flush=True)
    return mse_avg


# ============================================================
# 主函数
# ============================================================
def main():
    t_total = time.time()
    print("=" * 60, flush=True)
    print("  传统ML极致优化 V2 (快速版)", flush=True)
    print("  模型: XGB + LGBM + CatBoost + Ridge", flush=True)
    print("=" * 60, flush=True)

    X_raw, y_raw, labels = load_data()
    folds = get_folds()
    results = {}

    kw_default = dict(windows=(5,10,20), cross_ops=("mul","sub","div"), y_lags=20, add_ewm=True)

    # ---- f62 (残差) ----
    print(f"\n{'='*60}", flush=True)
    print(f"  f62 残差建模 | 基线: {BASELINE['f62']}", flush=True)
    print(f"{'='*60}", flush=True)
    # f62已经很好，只跑全局Stacking验证
    mse = run_global_stacking(X_raw, y_raw[:,0], labels, folds, "f62",
                              [(F54,F22),(F50,F22)], residual_col=F01, **kw_default)
    results["f62_global_stack"] = mse

    # ---- f63 ----
    print(f"\n{'='*60}", flush=True)
    print(f"  f63 | 基线: {BASELINE['f63']}", flush=True)
    print(f"{'='*60}", flush=True)
    cross_63 = [(F54,F22),(F54,F50),(F22,F50),(F54,F51),(F22,F48)]

    mse = run_global_stacking(X_raw, y_raw[:,1], labels, folds, "f63", cross_63, **kw_default)
    results["f63_global_stack"] = mse

    mse = run_mode_stacking(X_raw, y_raw[:,1], labels, folds, "f63", cross_63, **kw_default)
    results["f63_mode_stack"] = mse

    mse = run_model_avg(X_raw, y_raw[:,1], labels, folds, "f63", cross_63, **kw_default)
    results["f63_model_avg"] = mse

    # ---- f64 ----
    print(f"\n{'='*60}", flush=True)
    print(f"  f64 | 基线: {BASELINE['f64']}", flush=True)
    print(f"{'='*60}", flush=True)
    cross_64 = [(F50,F52),(F50,F53),(F52,F53),(F50,F48),(F50,F51),(F51,F52),(F51,F53)]

    mse = run_global_stacking(X_raw, y_raw[:,2], labels, folds, "f64", cross_64, **kw_default)
    results["f64_global_stack"] = mse

    mse = run_mode_stacking(X_raw, y_raw[:,2], labels, folds, "f64", cross_64, **kw_default)
    results["f64_mode_stack"] = mse

    mse = run_model_avg(X_raw, y_raw[:,2], labels, folds, "f64", cross_64, **kw_default)
    results["f64_model_avg"] = mse

    # ---- 汇总 ----
    print(f"\n{'='*60}", flush=True)
    print(f"  结果汇总", flush=True)
    print(f"{'='*60}", flush=True)
    for key in sorted(results.keys()):
        mse = results[key]
        t = key[:3]
        bl = BASELINE.get(t, 1e-8)
        pct = (mse - bl) / bl * 100
        flag = " ✅" if pct < 0 else ""
        print(f"  {key:<25s} MSE={mse:.8f} ({pct:+.1f}%){flag}", flush=True)

    # 总MSE
    best_f62 = min(v for k,v in results.items() if k.startswith("f62"))
    best_f63 = min(v for k,v in results.items() if k.startswith("f63"))
    best_f64 = min(v for k,v in results.items() if k.startswith("f64"))
    total = best_f62 + best_f63 + best_f64
    old_total = sum(BASELINE.values())
    print(f"\n  最佳总MSE: {total:.6f} (基线: {old_total:.6f}, 提升: {(total-old_total)/old_total*100:+.1f}%)", flush=True)

    # 保存
    df = pd.DataFrame([{"scheme": k, "mse": v} for k,v in results.items()])
    df.to_csv(os.path.join(OUTPUT_DIR, "extreme_ml_summary.csv"), index=False, float_format="%.8f")

    # 图
    fig, ax = plt.subplots(figsize=(10, 5))
    schemes = list(results.keys())
    mses = [results[s] for s in schemes]
    colors = ["#4CAF50" if results[s] < BASELINE.get(s[:3], 1e8) else "#E53935" for s in schemes]
    ax.barh(range(len(schemes)), mses, color=colors, edgecolor="black", linewidth=0.5)
    for t, bl in BASELINE.items():
        ax.axvline(bl, color="blue", linestyle="--", alpha=0.5, label=f"{t}基线={bl:.4f}" if t=="f62" else "")
    ax.set_yticks(range(len(schemes))); ax.set_yticklabels(schemes, fontsize=9)
    ax.set_xlabel("MSE"); ax.set_title("传统ML极致优化", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, m in enumerate(mses):
        ax.text(m*1.001, i, f"{m:.6f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "extreme_ml_comparison.png"), dpi=150); plt.close()

    print(f"\n[完成] 耗时 {time.time()-t_total:.0f}s", flush=True)
    print(f"[保存] {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()