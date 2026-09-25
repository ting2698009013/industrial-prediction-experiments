from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_COLS_34 = np.array([
    1,2,3,7,8,9,10,12,16,18,19,20,21,22,25,26,27,28,29,30,31,37,47,48,
    50,51,52,53,54,55,57,58,59,60
], dtype=np.int32)
SELECTED_PATH = PROJECT_ROOT / "configs" / "selected_feature_indices.npy"
try:
    SELECTED_COLS = np.load(SELECTED_PATH).astype(np.int32)
except Exception:
    SELECTED_COLS = DEFAULT_SELECTED_COLS_34

# 原始三列目标；历史输入仍然可使用可见段 f62/f63/f64
TARGET_COLS = np.array([62, 63, 64], dtype=np.int32)
# 本版 DL 只学习 f63/f64，f62 留给 ML/hybrid 分支
DL_TARGET_COLS = np.array([63, 64], dtype=np.int32)
DL_TARGET_NAMES = ["f63", "f64"]

BATCH_LEN = 60000
VISIBLE_RATIO = 0.8
N_BATCHES = 10
N_MODES = 3

KNOWN_SEGMENTS = [
    {"start": 0, "end": 108037, "label": 2},
    {"start": 108037, "end": 304397, "label": 0},
    {"start": 304397, "end": 600000, "label": 1},
]
