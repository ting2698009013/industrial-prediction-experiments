from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_COLS_34 = np.array([1,2,3,7,8,9,10,12,16,18,19,20,21,22,25,26,27,28,29,30,31,37,47,48,50,51,52,53,54,55,57,58,59,60], dtype=np.int32)
SELECTED_FEATURE_PATH = PROJECT_ROOT / 'configs' / 'selected_feature_indices.npy'
SELECTED_COLS = np.load(SELECTED_FEATURE_PATH).astype(np.int32) if SELECTED_FEATURE_PATH.exists() else DEFAULT_SELECTED_COLS_34
TARGET_COLS = np.array([62,63,64], dtype=np.int32)
BATCH_LEN=60000; VISIBLE_LEN=48000; PRED_LEN=12000; N_BATCHES=10
MODE_COLS_CORE=np.array([13,28,29,33,34,36,38,61], dtype=np.int32)
MODE_COLS_EXTENDED=np.array([13,28,29,33,34,36,38,61,3,60,16,31,9,19,32], dtype=np.int32)
KNOWN_SEGMENTS=[{'start':0,'end':108037,'label':2},{'start':108037,'end':304397,'label':0},{'start':304397,'end':600000,'label':1}]
F64_PAIRS=[(50,52),(50,53),(52,53),(50,48),(50,51),(51,52),(51,53)]
F63_PAIRS=[(54,22),(54,50),(22,50),(54,51),(22,48)]
