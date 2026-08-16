"""MoE-Mamba — sparse Mixture-of-Experts interleaved with selective-SSM blocks.

Motivation for security alert triage
------------------------------------
The corpora are extremely heterogeneous: GUIDE spans 8,428 distinct detectors,
BOTSv3 has 107 sourcetypes whose fields mean completely different things
(a DNS reply code and a Windows EventCode share no semantics). A single dense
FFN must model all of them with the same parameters. A sparse MoE lets the
router send each alert to the experts that specialise in its telemetry family,
while the shared Mamba blocks still model the field sequence.

Architecture (per layer)
------------------------
    x -> BiMamba(selective SSM)  ->  MoE-FFN(top-k of E experts)  -> x
Routing is token-level (each field token is routed independently), which lets
different FIELDS of the same alert use different experts.

Auxiliary load-balancing loss (Shazeer et al. / Switch Transformer):
    L_aux = E * sum_e  frac_tokens_to_e * mean_router_prob_e
keeps the router from collapsing onto one expert.

Expert utilisation is recorded so the routing map itself becomes an
interpretability artefact (which detectors / sourcetypes share an expert).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import BiMamba


class Expert(nn.Module):
    """A single feed-forward expert."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    """Token-level top-k sparse mixture of experts."""

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout)
                                      for _ in range(n_experts)])
        self.norm = nn.LayerNorm(d_model)
        # running utilisation counter (for the interpretability figure)
        self.register_buffer("util", torch.zeros(n_experts), persistent=False)

    def forward(self, x):                       # x: [B,L,d]
        B, L, d = x.shape
        h = self.norm(x)
        flat = h.reshape(-1, d)                 # [N,d], N = B*L tokens
        logits = self.router(flat)              # [N,E]
        probs = logits.softmax(-1)

        topv, topi = probs.topk(self.top_k, dim=-1)         # [N,k]
        topv = topv / topv.sum(-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(flat)
        # dispatch: run each expert once over the tokens routed to it
        for e in range(self.n_experts):
            mask = (topi == e)                              # [N,k]
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            w = topv[tok_idx, slot].unsqueeze(-1)
            out.index_add_(0, tok_idx, self.experts[e](flat[tok_idx]) * w)

        # load-balancing auxiliary loss
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device)
            counts.index_add_(0, topi.reshape(-1),
                              torch.ones(topi.numel(), device=x.device))
            self.util = 0.9 * self.util + 0.1 * (counts / counts.sum().clamp(min=1))
        frac = torch.zeros(self.n_experts, device=x.device)
        frac.index_add_(0, topi.reshape(-1),
                        torch.ones(topi.numel(), device=x.device))
        frac = frac / frac.sum().clamp(min=1)
        aux = self.n_experts * (frac * probs.mean(0)).sum()

        return x + out.view(B, L, d), aux


class MoEMambaTab(nn.Module):
    """Field-token sequence -> [BiMamba -> SparseMoE] x N -> pooled -> logits."""

    def __init__(self, n_cat_fields: int, cat_sizes: Sequence[int],
                 n_num_fields: int, n_classes: int = 3, d_model: int = 64,
                 n_layers: int = 2, d_state: int = 4, dropout: float = 0.1,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_cat, self.n_num = n_cat_fields, n_num_fields
        self.L = n_cat_fields + n_num_fields
        self.n_experts, self.top_k = n_experts, top_k

        offs, tot = [], 0
        for s in cat_sizes:
            offs.append(tot); tot += s
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.long))
        self.cat_emb = nn.Embedding(tot + 1, d_model)
        self.num_proj = nn.Linear(1, d_model)
        self.field_emb = nn.Embedding(max(1, self.L), d_model)

        self.mamba = nn.ModuleList([BiMamba(d_model, d_state, dropout)
                                    for _ in range(n_layers)])
        self.moe = nn.ModuleList([SparseMoE(d_model, n_experts, top_k,
                                            dropout=dropout)
                                  for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes))
        self.last_aux = torch.tensor(0.0)

    def forward(self, xc, xn, return_routing: bool = False):
        toks = []
        if self.n_cat:
            toks.append(self.cat_emb(xc + self.offsets))
        if self.n_num:
            toks.append(self.num_proj(xn.unsqueeze(-1)))
        x = torch.cat(toks, dim=1)
        x = x + self.field_emb(torch.arange(self.L, device=x.device))[None]

        aux_total = x.new_zeros(())
        routing = []
        for mb, mo in zip(self.mamba, self.moe):
            x = mb(x)
            if return_routing:
                with torch.no_grad():
                    h = mo.norm(x).reshape(-1, x.shape[-1])
                    routing.append(mo.router(h).softmax(-1)
                                   .view(x.shape[0], x.shape[1], -1).cpu())
            x, aux = mo(x)
            aux_total = aux_total + aux
        self.last_aux = aux_total / max(1, len(self.moe))

        x = self.norm(x)
        logits = self.head(torch.cat([x.mean(1), x.max(1).values], dim=-1))
        return (logits, routing) if return_routing else logits

    def expert_utilisation(self) -> np.ndarray:
        """[n_layers, n_experts] EMA of the fraction of tokens each expert got."""
        return np.stack([m.util.detach().cpu().numpy() for m in self.moe])

    def active_params(self) -> int:
        """Params actually used per token (top_k of n_experts), vs total."""
        total = sum(p.numel() for p in self.parameters())
        per_expert = sum(p.numel() for p in self.moe[0].experts[0].parameters())
        inactive = len(self.moe) * (self.n_experts - self.top_k) * per_expert
        return total - inactive
