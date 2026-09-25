from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import SELECTED_COLS, TARGET_COLS, DL_TARGET_COLS, BATCH_LEN, VISIBLE_RATIO, N_BATCHES, N_MODES
from .utils import mode_switch_features, time_features

@dataclass
class SampleConfig:
    history_len: int
    pred_len: int
    cut_stride: int = 512
    random_jitter: int = 0
    min_cut: int | None = None


def _as_flat_labels(labels, n):
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = labels.reshape(-1)
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != data length {n}")
    return labels.astype(np.int16)


def build_input_matrix(data, labels, selected_cols=SELECTED_COLS, zero_future_target_mask=None, near_switch_win=512):
    """构建历史/未来输入。

    hist = selected process features + 可见目标 f62/f63/f64 + mode onehot/switch features
    fut  = selected process features + mode onehot/switch features

    zero_future_target_mask 只影响 hist 中的目标列，方便模拟官方 question：预测窗口目标置 0。
    现有 sliding-cut fold 训练默认不需要传 mask；final/question-style 训练可传 batch 后 20% mask。
    """
    n = len(data)
    labels = _as_flat_labels(labels, n)
    proc = np.asarray(data[:, selected_cols], dtype=np.float32)
    targets_hist = np.asarray(data[:, TARGET_COLS], dtype=np.float32).copy()
    if zero_future_target_mask is not None:
        m = np.asarray(zero_future_target_mask, dtype=bool)
        if len(m) != n:
            raise ValueError(f"zero mask length {len(m)} != data length {n}")
        targets_hist[m, :] = 0.0
    dl_targets = np.asarray(data[:, DL_TARGET_COLS], dtype=np.float32)
    if labels is None:
        mode = np.zeros((n, N_MODES + 2), dtype=np.float32)
    else:
        mode = mode_switch_features(labels, near_win=near_switch_win, n_modes=N_MODES)
    hist = np.concatenate([proc, targets_hist, mode], axis=1).astype(np.float32)
    fut = np.concatenate([proc, mode], axis=1).astype(np.float32)
    return hist, fut, dl_targets


def make_question_style_zero_mask(n, batch_len=BATCH_LEN, visible_ratio=VISIBLE_RATIO):
    visible_len = int(round(batch_len * visible_ratio))
    mask = np.zeros(n, dtype=bool)
    nb = n // batch_len
    for bi in range(nb):
        s = bi * batch_len
        mask[s + visible_len : s + batch_len] = True
    return mask


class CutSequenceDataset(Dataset):
    """沿用 v1 sliding-cut 训练方式：cut 之前为历史，cut 之后 pred_len 为监督。"""
    def __init__(self, data, labels, sample_cfg, train_end, input_scaler=None, target_scaler=None, selected_cols=SELECTED_COLS, zero_future_target_mask=None, near_switch_win=512):
        self.data, self.labels, self.cfg = data, labels, sample_cfg
        self.train_end, self.selected_cols = int(train_end), selected_cols
        self.hist_all, self.fut_all, self.targets_all = build_input_matrix(data, labels, selected_cols, zero_future_target_mask=zero_future_target_mask, near_switch_win=near_switch_win)
        self.n_proc = len(selected_cols)
        if input_scaler is not None:
            self.hist_all = self.hist_all.copy(); self.fut_all = self.fut_all.copy()
            # 只缩放 selected process + 3 个历史目标；mode/switch 特征不缩放
            self.hist_all[:, :self.n_proc+3] = input_scaler.transform_np(self.hist_all[:, :self.n_proc+3])
            dummy = np.concatenate([self.fut_all[:, :self.n_proc], np.zeros((len(self.fut_all),3), dtype=np.float32)], axis=1)
            self.fut_all[:, :self.n_proc] = input_scaler.transform_np(dummy)[:, :self.n_proc]
        self.targets_scaled = target_scaler.transform_np(self.targets_all) if target_scaler is not None else self.targets_all.astype(np.float32)

        min_cut = sample_cfg.min_cut if sample_cfg.min_cut is not None else sample_cfg.history_len
        max_cut = self.train_end - sample_cfg.pred_len
        if max_cut <= min_cut:
            raise ValueError(f"not enough train history: min_cut={min_cut}, max_cut={max_cut}")
        self.cuts = np.arange(min_cut, max_cut+1, sample_cfg.cut_stride, dtype=np.int64)

    def __len__(self): return len(self.cuts)

    def __getitem__(self, idx):
        cut = int(self.cuts[idx])
        if self.cfg.random_jitter:
            j = np.random.randint(-self.cfg.random_jitter, self.cfg.random_jitter+1)
            cut = int(np.clip(cut+j, self.cfg.history_len, self.train_end-self.cfg.pred_len))
        hs, he = cut - self.cfg.history_len, cut
        fs, fe = cut, cut + self.cfg.pred_len
        hist = self.hist_all[hs:he]
        fut = np.concatenate([self.fut_all[fs:fe], time_features(self.cfg.pred_len)], axis=1).astype(np.float32)
        y = self.targets_scaled[fs:fe].astype(np.float32)
        return {"hist": torch.from_numpy(hist), "future": torch.from_numpy(fut), "target": torch.from_numpy(y)}


class OfficialBatchWindowDataset(Dataset):
    """官方 question-style 训练样本：每个 batch 前 80% 可见，后 20% 作为监督窗口。"""
    def __init__(self, data, labels, history_len, input_scaler=None, target_scaler=None, selected_cols=SELECTED_COLS, batch_len=BATCH_LEN, visible_ratio=VISIBLE_RATIO, near_switch_win=512):
        self.data = data
        self.history_len = int(history_len)
        self.batch_len = int(batch_len)
        self.visible_len = int(round(batch_len * visible_ratio))
        self.pred_len = self.batch_len - self.visible_len
        self.n_batches = len(data) // self.batch_len
        zmask = make_question_style_zero_mask(len(data), self.batch_len, visible_ratio)
        self.hist_all, self.fut_all, self.targets_all = build_input_matrix(data, labels, selected_cols, zero_future_target_mask=zmask, near_switch_win=near_switch_win)
        self.n_proc = len(selected_cols)
        if input_scaler is not None:
            self.hist_all = self.hist_all.copy(); self.fut_all = self.fut_all.copy()
            self.hist_all[:, :self.n_proc+3] = input_scaler.transform_np(self.hist_all[:, :self.n_proc+3])
            dummy = np.concatenate([self.fut_all[:, :self.n_proc], np.zeros((len(self.fut_all),3), dtype=np.float32)], axis=1)
            self.fut_all[:, :self.n_proc] = input_scaler.transform_np(dummy)[:, :self.n_proc]
        self.targets_scaled = target_scaler.transform_np(self.targets_all) if target_scaler is not None else self.targets_all.astype(np.float32)
        if self.history_len > self.visible_len:
            raise ValueError(f"history_len={self.history_len} cannot exceed visible_len={self.visible_len}")

    def __len__(self): return self.n_batches

    def __getitem__(self, bi):
        s = int(bi) * self.batch_len
        cut = s + self.visible_len
        end = s + self.batch_len
        hs = cut - self.history_len
        hist = self.hist_all[hs:cut]
        fut = np.concatenate([self.fut_all[cut:end], time_features(end-cut)], axis=1).astype(np.float32)
        y = self.targets_scaled[cut:end].astype(np.float32)
        return {"hist": torch.from_numpy(hist), "future": torch.from_numpy(fut), "target": torch.from_numpy(y), "batch": int(bi)}


def make_eval_sample(data, labels, cut, end, history_len, input_scaler=None, zero_future_target_mask=None, near_switch_win=512):
    if cut - history_len < 0:
        raise ValueError("not enough history for eval sample")
    pred_len = end - cut
    hist_all, fut_all, _ = build_input_matrix(data, labels, zero_future_target_mask=zero_future_target_mask, near_switch_win=near_switch_win)
    n_proc = len(SELECTED_COLS)
    if input_scaler is not None:
        hist_all = hist_all.copy(); fut_all = fut_all.copy()
        hist_all[:, :n_proc+3] = input_scaler.transform_np(hist_all[:, :n_proc+3])
        dummy = np.concatenate([fut_all[:, :n_proc], np.zeros((len(fut_all),3), dtype=np.float32)], axis=1)
        fut_all[:, :n_proc] = input_scaler.transform_np(dummy)[:, :n_proc]
    hist = hist_all[cut-history_len:cut]
    fut = np.concatenate([fut_all[cut:end], time_features(pred_len)], axis=1).astype(np.float32)
    return {"hist": torch.from_numpy(hist[None]), "future": torch.from_numpy(fut[None])}
