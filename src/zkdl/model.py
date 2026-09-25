import torch
from torch import nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.net(x) + self.down(x)

class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, levels=4, kernel_size=5, dropout=0.1):
        super().__init__()
        blocks, ch = [], in_dim
        for i in range(levels):
            blocks.append(TemporalBlock(ch, hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
            ch = hidden
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        z = self.net(x.transpose(1,2))
        return z.transpose(1,2)

class DualHeadTCNForecaster(nn.Module):
    """共享 encoder，f63/f64 两个独立 head。输出 shape=(B,T,2)，顺序为 f63,f64。"""
    def __init__(self, hist_dim, future_dim, hidden=128, levels=4, dropout=0.1):
        super().__init__()
        self.hist_encoder = TCNEncoder(hist_dim, hidden, levels, dropout=dropout)
        self.future_encoder = TCNEncoder(future_dim, hidden, levels, dropout=dropout)
        self.context_proj = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.shared = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head63 = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.head64 = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))

    def forward(self, hist, future):
        hz = self.hist_encoder(hist)
        ctx = self.context_proj(torch.cat([hz[:, -1], hz.mean(dim=1)], dim=-1))
        fz = self.future_encoder(future)
        ctx = ctx[:, None, :].expand(-1, fz.shape[1], -1)
        h = self.shared(torch.cat([fz, ctx], dim=-1))
        y63 = self.head63(h)
        y64 = self.head64(h)
        return torch.cat([y63, y64], dim=-1)

# 兼容旧脚本 import 名称，但现在已是双 head、只输出 f63/f64
DirectTCNForecaster = DualHeadTCNForecaster
