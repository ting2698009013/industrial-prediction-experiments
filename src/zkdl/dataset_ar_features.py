from __future__ import annotations

import numpy as np
import torch

from .config import SELECTED_COLS
from .dataset import OfficialBatchWindowDataset


def _shift_past(y, lag):
    y = np.asarray(y, dtype=np.float32)
    out = np.empty_like(y)
    out[:lag] = y[0]
    out[lag:] = y[:-lag]
    return out


def _rolling_mean_std(y, window):
    """
    trailing rolling mean/std.
    第 t 点只使用 [max(0,t-window+1), t]，不看未来。
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)

    cs = np.concatenate([[0.0], np.cumsum(y, dtype=np.float64)])
    cs2 = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])

    idx = np.arange(n)
    st = np.maximum(0, idx - window + 1)
    ed = idx + 1
    cnt = ed - st

    s = cs[ed] - cs[st]
    s2 = cs2[ed] - cs2[st]

    mean = s / cnt
    var = s2 / cnt - mean * mean
    var = np.maximum(var, 0.0)

    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _ewm(y, alpha):
    y = np.asarray(y, dtype=np.float32)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def _slope_past(y, window):
    y = np.asarray(y, dtype=np.float32)
    lag = _shift_past(y, window)
    denom = np.minimum(np.arange(len(y)), window).astype(np.float32)
    denom[denom < 1.0] = 1.0
    return ((y - lag) / denom).astype(np.float32)


def _one_target_ar_features(y):
    """
    输入 y 是已经标准化后的 f63 或 f64 历史序列。
    返回:
      hist_feats:   (history_len, n_hist_ar)
      future_state: (n_future_ar,)
    """
    y = np.asarray(y, dtype=np.float32)

    lag1 = _shift_past(y, 1)
    lag2 = _shift_past(y, 2)
    lag4 = _shift_past(y, 4)

    diff1 = y - lag1
    diff2 = lag1 - lag2

    mean64, std64 = _rolling_mean_std(y, 64)
    mean256, std256 = _rolling_mean_std(y, 256)
    mean1024, std1024 = _rolling_mean_std(y, 1024)

    ewm_fast = _ewm(y, 0.10)
    ewm_slow = _ewm(y, 0.01)

    slope64 = _slope_past(y, 64)
    slope256 = _slope_past(y, 256)

    last_minus_mean64 = y - mean64
    last_minus_mean256 = y - mean256
    mean64_minus_mean1024 = mean64 - mean1024

    hist_feats = np.stack(
        [
            lag1,
            lag2,
            lag4,
            diff1,
            diff2,
            mean64,
            mean256,
            mean1024,
            std64,
            std256,
            std1024,
            ewm_fast,
            ewm_slow,
            slope64,
            slope256,
            last_minus_mean64,
            last_minus_mean256,
            mean64_minus_mean1024,
        ],
        axis=1,
    ).astype(np.float32)

    future_state = np.asarray(
        [
            y[-1],
            lag1[-1],
            lag2[-1],
            diff1[-1],
            diff2[-1],
            mean64[-1],
            mean256[-1],
            mean1024[-1],
            std64[-1],
            std256[-1],
            std1024[-1],
            ewm_fast[-1],
            ewm_slow[-1],
            slope64[-1],
            slope256[-1],
            last_minus_mean64[-1],
            last_minus_mean256[-1],
            mean64_minus_mean1024[-1],
        ],
        dtype=np.float32,
    )

    return hist_feats, future_state


def append_ar_features(hist, future, n_proc=None):
    """
    hist:
      baseline hist, shape = (history_len, hist_dim)
      结构:
        selected process features + f62/f63/f64 + mode features

    future:
      baseline future, shape = (pred_len, future_dim)
      结构:
        selected process features + mode features + time features

    这里使用 hist 中已经标准化后的 f63/f64 计算 AR 特征。
    """
    hist = np.asarray(hist, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)

    if n_proc is None:
        n_proc = len(SELECTED_COLS)

    f63_idx = n_proc + 1
    f64_idx = n_proc + 2

    y63 = hist[:, f63_idx]
    y64 = hist[:, f64_idx]

    hist63, state63 = _one_target_ar_features(y63)
    hist64, state64 = _one_target_ar_features(y64)

    hist_ar = np.concatenate([hist, hist63, hist64], axis=1).astype(np.float32)

    state = np.concatenate([state63, state64], axis=0).astype(np.float32)
    state_rep = np.repeat(state[None, :], repeats=len(future), axis=0)

    future_ar = np.concatenate([future, state_rep], axis=1).astype(np.float32)

    return hist_ar, future_ar


class OfficialBatchWindowDatasetARFeatures(OfficialBatchWindowDataset):
    """
    独立 AR 特征增强版 Dataset。

    不改原始 OfficialBatchWindowDataset。
    只在当前 direct_ar_features 实验中使用。

    增强内容：
      hist 追加 f63/f64 的 lag、rolling、ewm、slope、std 等逐点 AR 特征
      future 追加从可见历史末端提取的 AR state，并复制到每个预测点
    """
    def __getitem__(self, bi):
        sample = super().__getitem__(bi)

        hist = sample["hist"].cpu().numpy()
        future = sample["future"].cpu().numpy()

        hist_ar, future_ar = append_ar_features(
            hist=hist,
            future=future,
            n_proc=self.n_proc,
        )

        sample["hist"] = torch.from_numpy(hist_ar)
        sample["future"] = torch.from_numpy(future_ar)

        return sample