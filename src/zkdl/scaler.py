import numpy as np

class Standardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-8] = 1.0

    @classmethod
    def fit(cls, x):
        x = x.astype(np.float64)
        return cls(np.nanmean(x, axis=0), np.nanstd(x, axis=0))

    def transform_np(self, x):
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_np(self, x):
        return (x.astype(np.float32) * self.std + self.mean).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])
