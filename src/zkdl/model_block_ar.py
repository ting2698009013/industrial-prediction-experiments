import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, levels=4, kernel_size=5, dropout=0.1):
        super().__init__()
        blocks, ch = [], in_dim
        for i in range(levels):
            blocks.append(
                TemporalBlock(
                    ch,
                    hidden,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            ch = hidden
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return z.transpose(1, 2)


class DualHeadTCNForecaster(nn.Module):
    """
    Direct baseline:
    shared encoder + f63/f64 dual head.
    输入:
      hist:   (B, H, hist_dim)
      future: (B, T, future_dim)
    输出:
      y:      (B, T, 2), 顺序 f63, f64
    """
    def __init__(self, hist_dim, future_dim, hidden=128, levels=4, dropout=0.1):
        super().__init__()
        self.hist_encoder = TCNEncoder(hist_dim, hidden, levels, dropout=dropout)
        self.future_encoder = TCNEncoder(future_dim, hidden, levels, dropout=dropout)

        self.context_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head63 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.head64 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, hist, future):
        hz = self.hist_encoder(hist)
        ctx = self.context_proj(torch.cat([hz[:, -1], hz.mean(dim=1)], dim=-1))

        fz = self.future_encoder(future)
        ctx = ctx[:, None, :].expand(-1, fz.shape[1], -1)

        h = self.shared(torch.cat([fz, ctx], dim=-1))

        y63 = self.head63(h)
        y64 = self.head64(h)
        return torch.cat([y63, y64], dim=-1)


class BlockARDualHeadTCNForecaster(nn.Module):
    """
    Block autoregressive dual-head model.

    和 Direct 模型接口一致:
      forward(hist, future) -> (B, T, 2)

    区别:
      future 被切成多个 block。
      每个 block 的预测会使用上一个 block 的最后预测值作为 AR 状态。
      初始 AR 状态来自 hist 最后一个可见时刻的 f63/f64。

    注意:
      这里不是逐点递推，而是 block-level autoregressive。
    """
    def __init__(
        self,
        hist_dim,
        future_dim,
        hidden=128,
        levels=4,
        dropout=0.1,
        block_len=256,
        ar_target_indices=None,
        detach_ar=True,
    ):
        super().__init__()

        self.block_len = int(block_len)
        self.detach_ar = bool(detach_ar)

        if ar_target_indices is None:
            raise ValueError("BlockARDualHeadTCNForecaster 需要 ar_target_indices=(f63_index, f64_index)")
        self.ar_target_indices = tuple(int(x) for x in ar_target_indices)

        self.hist_encoder = TCNEncoder(hist_dim, hidden, levels, dropout=dropout)
        self.future_encoder = TCNEncoder(future_dim, hidden, levels, dropout=dropout)

        self.context_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ar_proj = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head63 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self.head64 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, hist, future):
        """
        hist:   (B, H, hist_dim)
        future: (B, T, future_dim)
        return: (B, T, 2)
        """
        bsz, total_t, _ = future.shape

        hz = self.hist_encoder(hist)
        ctx = self.context_proj(torch.cat([hz[:, -1], hz.mean(dim=1)], dim=-1))

        # 初始 AR 状态：最后一个可见时刻的 f63/f64，注意这里已经是 scaler 后的空间
        prev_y = hist[:, -1, list(self.ar_target_indices)]

        outs = []

        for st in range(0, total_t, self.block_len):
            ed = min(st + self.block_len, total_t)
            fut_block = future[:, st:ed, :]

            fz = self.future_encoder(fut_block)

            block_t = ed - st
            ctx_rep = ctx[:, None, :].expand(-1, block_t, -1)

            ar_h = self.ar_proj(prev_y)
            ar_rep = ar_h[:, None, :].expand(-1, block_t, -1)

            h = self.shared(torch.cat([fz, ctx_rep, ar_rep], dim=-1))

            y63 = self.head63(h)
            y64 = self.head64(h)
            y = torch.cat([y63, y64], dim=-1)

            outs.append(y)

            prev_y = y[:, -1, :]
            if self.detach_ar:
                prev_y = prev_y.detach()

        return torch.cat(outs, dim=1)


DirectTCNForecaster = DualHeadTCNForecaster