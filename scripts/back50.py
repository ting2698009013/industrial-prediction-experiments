import numpy as np
from pathlib import Path

# 原训练集
train_path = Path("data\processed\clean_train_wcj.npy")
mode_path  = Path("runs\modes_train\mode_labels.npy")

pseudo_train = np.load(train_path)
pseudo_mode  = np.load(mode_path)

# 后 50%
N = len(pseudo_train)
half_idx = N // 2

pseudo_train_back50 = pseudo_train[half_idx:]
pseudo_mode_back50  = pseudo_mode[half_idx:]

# 保存
out_dir = Path("data/back50")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "pseudo_train.npy", pseudo_train_back50)
np.save(out_dir / "pseudo_train_mode_labels.npy", pseudo_mode_back50)

print(f"Saved pseudo_train back 50%: {pseudo_train_back50.shape}")
print(f"Saved pseudo_train_mode_labels back 50%: {pseudo_mode_back50.shape}")