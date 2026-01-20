#src/pos_encoding.py
import math
import torch
import torch.nn as nn

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)  # (max_len, d_model)

    def forward(self, t_idx: torch.Tensor) -> torch.Tensor:
        """
        t_idx: (...,) int64
        returns: (..., d_model)
        """
        return self.pe[t_idx]