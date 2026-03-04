# src/models/enc_dec/enc_dec.py
import torch
import torch.nn as nn

from src.pos_encoding import SinusoidalTimeEncoding


class EncoderDecoderBaseline(nn.Module):
    """
    Wayformer-style:
      - Encoder builds 'memory' from tokens
      - M learnable queries go through TransformerDecoder (cross-attn to memory)
      - Output trajectories: (B, M, Tf, 2)
      - Output scores/logits: (B, M)  (optional)
    """

    def __init__(
        self,
        T=10, Tf=25, K=8,
        ego_dim: int = 28,
        nb_dim: int = 16,
        use_neighbors: bool = True,
        use_slot_emb: bool = True,
        d_model=128, nhead=4,
        enc_layers=2, dec_layers=2,
        dropout=0.1,
        predict_delta: bool = False,
        M: int = 6,
        return_scores: bool = True,
    ):
        super().__init__()
        self.T, self.Tf, self.K = T, Tf, K
        self.use_neighbors = use_neighbors
        self.use_slot_emb = use_slot_emb
        self.predict_delta = predict_delta

        self.M = M
        self.return_scores = return_scores

        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.nb_proj = nn.Linear(nb_dim, d_model)

        self.time_enc = SinusoidalTimeEncoding(d_model, max_len=T)
        self.slot_emb = nn.Embedding(1 + K, d_model) if use_slot_emb else None

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # Learnable queries (modes)
        self.query_emb = nn.Embedding(M, d_model)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # Heads
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, Tf * 2),
        )
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, x_ego, x_nb, nb_mask, style_prob=None, style_valid=None):
        """
        Keep signature compatible with TransformerStyleBaseline calls.
        style_prob/style_valid are ignored here.
        """
        B, T, _ = x_ego.shape
        ego_tok = self.ego_proj(x_ego)  # (B,T,d)

        if self.use_neighbors:
            nb_tok = self.nb_proj(x_nb)  # (B,T,K,d)
            valid_any = nb_mask.any(dim=-1)

            nb_tok = nb_tok * valid_any.unsqueeze(-1).unsqueeze(-1)

            tok = torch.cat([ego_tok.unsqueeze(2), nb_tok], dim=2)  # (B,T,1+K,d)
            tok = tok.reshape(B, T * (1 + self.K), -1)

            valid = torch.ones((B, T, 1 + self.K), device=tok.device, dtype=torch.bool)
            valid[:, :, 1:] = nb_mask
            key_padding_mask = ~valid.reshape(B, -1)  # True=pad
            key_padding_mask[:, 0] = False
        else:
            tok = ego_tok
            key_padding_mask = None

        # time PE
        if self.use_neighbors:
            t_idx = torch.arange(T, device=tok.device).repeat_interleave(1 + self.K)
        else:
            t_idx = torch.arange(T, device=tok.device)
        
        tok = tok + self.time_enc(t_idx).unsqueeze(0).repeat(B, 1, 1)

        # slot emb
        if self.use_neighbors and (self.slot_emb is not None):
            slot_ids = torch.arange(0, 1 + self.K, device=tok.device).repeat(T)
            tok = tok + self.slot_emb(slot_ids).unsqueeze(0).repeat(B, 1, 1)

        # encoder memory
        memory = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # (B,L,d)

        # decoder queries
        q = self.query_emb.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,M,d)
        dec_out = self.decoder(
            tgt=q,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )  # (B,M,d)

        traj = self.traj_head(dec_out).view(B, self.M, self.Tf, 2)  # (B,M,Tf,2)
        scores = self.score_head(dec_out).squeeze(-1)               # (B,M)

        return (traj, scores) if self.return_scores else traj