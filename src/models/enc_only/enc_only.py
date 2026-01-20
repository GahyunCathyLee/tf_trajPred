#src/models/enc_only.py
import torch
import torch.nn as nn

from src.pos_encoding import SinusoidalTimeEncoding

class TransformerBaseline(nn.Module):
    def __init__(
        self,
        T=15, Tf=25, K=8,
        ego_dim: int = 16,
        nb_dim: int = 9,
        use_neighbors: bool = True,
        use_slot_emb: bool = True,
        d_model=128, nhead=4, num_layers=2, dropout=0.1, predict_delta: bool = False,
    ):
        super().__init__()
        self.T, self.Tf, self.K = T, Tf, K
        self.use_neighbors = use_neighbors
        self.use_slot_emb = use_slot_emb

        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.nb_proj  = nn.Linear(nb_dim, d_model)

        self.predict_delta = predict_delta

        self.time_enc = SinusoidalTimeEncoding(d_model, max_len=T)
        if use_slot_emb:
            self.slot_emb = nn.Embedding(1 + K, d_model)
        else:
            self.slot_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, Tf * 2)
        )

    def forward(self, x_ego, x_nb, nb_mask, **kwargs):
        B, T, _ = x_ego.shape
        ego_tok = self.ego_proj(x_ego)  # (B,T,d)

        if self.use_neighbors:
            nb_tok = self.nb_proj(x_nb)  # (B,T,K,d)
            tok = torch.cat([ego_tok.unsqueeze(2), nb_tok], dim=2)  # (B,T,1+K,d)
            tok = tok.reshape(B, T*(1+self.K), -1)                  # (B,L,d)

            # masks
            valid = torch.ones((B, T, 1+self.K), device=tok.device, dtype=torch.bool)
            valid[:, :, 1:] = nb_mask
            key_padding_mask = ~valid.reshape(B, -1)
            last_ego_idx = (T-1)*(1+self.K)
        else:
            tok = ego_tok  # (B,T,d)
            key_padding_mask = None
            last_ego_idx = T-1

        # time pe
        if self.use_neighbors:
            t_idx = torch.arange(T, device=tok.device).repeat_interleave(1+self.K)
            time_pe = self.time_enc(t_idx).unsqueeze(0).repeat(B, 1, 1)
        else:
            t_idx = torch.arange(T, device=tok.device)
            time_pe = self.time_enc(t_idx).unsqueeze(0).repeat(B, 1, 1)

        tok = tok + time_pe

        # slot embedding (optional)
        if self.use_neighbors and self.slot_emb is not None:
            slot_ids = torch.arange(0, 1+self.K, device=tok.device).repeat(T)
            tok = tok + self.slot_emb(slot_ids).unsqueeze(0).repeat(B, 1, 1)

        H = self.encoder(tok, src_key_padding_mask=key_padding_mask)
        h = H[:, last_ego_idx, :]
        out = self.head(h).reshape(B, self.Tf, 2)
        return out
