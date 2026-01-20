#src/models/build.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import math
import sys
import torch
import torch.nn as nn

# -------------------------
# Scheduler
# -------------------------
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    sched_type: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    sched_type = (sched_type or "none").lower()

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if sched_type == "cosine":
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# -------------------------
# Model builder
# -------------------------
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    mcfg = cfg.get("model", {})
    name = str(mcfg.get("name", "encdec")).lower()

    if name in ("enc_dec", "encdecbaseline", "encdec"):
        try:
            from models.enc_dec import EncoderDecoderBaseline  # type: ignore
        except Exception:
            # fallback: if you kept old structure
            from src.models.enc_dec.enc_dec import EncoderDecoderBaseline  # type: ignore

        return EncoderDecoderBaseline(
            T=int(mcfg["T"]),
            Tf=int(mcfg["Tf"]),
            K=int(mcfg.get("K", 8)),
            ego_dim=int(mcfg["ego_dim"]),
            nb_dim=int(mcfg["nb_dim"]),
            use_neighbors=bool(mcfg.get("use_neighbors", True)),
            use_slot_emb=bool(mcfg.get("use_slot_emb", True)),
            d_model=int(mcfg.get("d_model", 128)),
            nhead=int(mcfg.get("nhead", 4)),
            enc_layers=int(mcfg.get("enc_layers", 2)),
            dec_layers=int(mcfg.get("dec_layers", 2)),
            dropout=float(mcfg.get("dropout", 0.1)),
            predict_delta=bool(mcfg.get("predict_delta", False)),
            M=int(mcfg.get("M", 6)),
            return_scores=bool(mcfg.get("return_scores", True)),
        )

    raise ValueError(f"Unknown model.name: {name}")