"""MambaTab — the shared tabular Mamba classifier used on GUIDE, AIT-ADS, BOTSv3.

A row is turned into a SEQUENCE of (field, value) tokens; a stack of
bidirectional selective-SSM blocks scans that sequence; mean+max pooling feeds a
linear classifier. Identical architecture across datasets so results are
comparable; only the field count / vocab sizes differ.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .mamba import BiMamba


class MambaTab(nn.Module):
    def __init__(self, n_cat_fields: int, cat_sizes, n_num_fields: int,
                 n_classes: int = 3, d_model: int = 64, n_layers: int = 2,
                 d_state: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_cat, self.n_num = n_cat_fields, n_num_fields
        self.L = n_cat_fields + n_num_fields

        offs, tot = [], 0
        for s in cat_sizes:
            offs.append(tot); tot += s
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.long))
        self.cat_emb = nn.Embedding(tot + 1, d_model)
        self.num_proj = nn.Linear(1, d_model)
        self.field_emb = nn.Embedding(self.L, d_model)
        self.blocks = nn.ModuleList([BiMamba(d_model, d_state, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))

    def forward(self, xc, xn):
        toks = []
        if self.n_cat:
            toks.append(self.cat_emb(xc + self.offsets))
        if self.n_num:
            toks.append(self.num_proj(xn.unsqueeze(-1)))
        x = torch.cat(toks, dim=1)
        x = x + self.field_emb(torch.arange(self.L, device=x.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(torch.cat([x.mean(1), x.max(1).values], dim=-1))


def prepare_tabular(records, cat_fields, num_fields, max_card: int = 200_000):
    """records (list of dicts) -> (Xc int64 [N,nc], Xn float32 [N,nn], cat_sizes)."""
    import pandas as pd
    df = pd.DataFrame(records)
    for c in cat_fields + num_fields:
        if c not in df.columns:
            df[c] = np.nan

    cat_arrs, cat_sizes = [], []
    for c in cat_fields:
        codes = pd.factorize(df[c].astype("string").fillna("__NA__"))[0].astype(np.int64)
        if codes.max() + 1 > max_card:
            codes = codes % max_card
        cat_arrs.append(codes)
        cat_sizes.append(int(codes.max()) + 2)
    Xc = (np.stack(cat_arrs, axis=1) if cat_arrs
          else np.zeros((len(df), 0), dtype=np.int64))

    if num_fields:
        Xn = np.stack([pd.to_numeric(df[c], errors="coerce").fillna(-1).to_numpy()
                       for c in num_fields], axis=1).astype(np.float32)
        mu, sd = Xn.mean(0, keepdims=True), Xn.std(0, keepdims=True) + 1e-6
        Xn = (Xn - mu) / sd
    else:
        Xn = np.zeros((len(df), 0), dtype=np.float32)
    return Xc, Xn, cat_sizes
