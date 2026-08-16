"""Federated aggregation strategies for MoE-Mamba.

FedAvg is included only as the baseline to beat. The strategies of interest are
the ones that correct for client drift under non-IID partitions, which is
exactly the regime here: clients are real organisations / scenarios / hosts,
so their label and feature distributions genuinely differ.

    FedAvg    weighted parameter mean (baseline)
    FedProx   FedAvg + local proximal term mu/2 ||w - w_global||^2
    FedAdam   server-side Adam over the pseudo-gradient (FedOpt family)
    SCAFFOLD  control variates that cancel client drift in the local update

Memory note: SCAFFOLD keeps one control variate per client plus one global,
i.e. (n_clients + 1) x model size. That is fine for the small tabular encoders
used on AIT-ADS and BOTSv3, but not for the 111M-parameter GUIDE model, where
the embedding table dominates; `scaffold_feasible` guards this.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def _zeros_like_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()}


def scaffold_feasible(n_params: int, n_clients: int, budget_gb: float = 6.0) -> bool:
    """(n_clients + 1) control-variate copies must fit in `budget_gb`."""
    bytes_needed = 4.0 * n_params * (n_clients + 1)
    return bytes_needed <= budget_gb * (1024 ** 3)


def weighted_average(states: Sequence[Dict[str, torch.Tensor]],
                     weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """Weighted mean of client parameter dicts (the FedAvg step)."""
    tot = float(sum(weights))
    out = {}
    for k in states[0]:
        ref = states[0][k]
        if not ref.is_floating_point():
            out[k] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += st[k].to(torch.float32) * (float(w) / tot)
        out[k] = acc.to(ref.dtype)
    return out


class ServerOptimizer:
    """Server-side adaptive aggregation (FedOpt / FedAdam, Reddi et al.).

    The aggregated client delta is treated as a pseudo-gradient and fed to an
    Adam-style update at the server, which damps the oscillation that plain
    parameter averaging shows under heterogeneous clients.
    """

    def __init__(self, global_state: Dict[str, torch.Tensor], lr: float = 1e-2,
                 beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-3):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = _zeros_like_state(global_state)
        self.v = _zeros_like_state(global_state)
        self.t = 0

    def step(self, global_state: Dict[str, torch.Tensor],
             new_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.t += 1
        out = {}
        for k, g in global_state.items():
            if not g.is_floating_point():
                out[k] = g.clone()
                continue
            delta = (new_state[k].to(torch.float32) - g.to(torch.float32))
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * delta
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * delta.pow(2)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = (g.to(torch.float32)
                      + self.lr * mhat / (vhat.sqrt() + self.eps)).to(g.dtype)
        return out


class ScaffoldState:
    """Control variates for SCAFFOLD (Karimireddy et al.).

    Local step becomes  w <- w - eta * (g_i - c_i + c), which removes the
    client-drift term that makes FedAvg stall on non-IID partitions.
    """

    def __init__(self, global_state: Dict[str, torch.Tensor], client_ids: List[str]):
        self.c = _zeros_like_state(global_state)
        self.c_i = {cid: _zeros_like_state(global_state) for cid in client_ids}

    def correction(self, cid: str) -> Dict[str, torch.Tensor]:
        """The (c - c_i) term added to each local gradient."""
        return {k: self.c[k] - self.c_i[cid][k] for k in self.c}

    def update_client(self, cid: str, global_state: Dict[str, torch.Tensor],
                      local_state: Dict[str, torch.Tensor],
                      n_steps: int, lr: float) -> Dict[str, torch.Tensor]:
        """c_i^+ = c_i - c + (w_global - w_local) / (n_steps * lr); returns delta c_i."""
        delta = {}
        denom = max(1e-12, n_steps * lr)
        for k in self.c:
            if not global_state[k].is_floating_point():
                continue
            new_ci = (self.c_i[cid][k] - self.c[k]
                      + (global_state[k].to(torch.float32)
                         - local_state[k].to(torch.float32)) / denom)
            delta[k] = new_ci - self.c_i[cid][k]
            self.c_i[cid][k] = new_ci
        return delta

    def update_server(self, deltas: List[Dict[str, torch.Tensor]],
                      n_total_clients: int) -> None:
        """c <- c + (1/N) sum_i delta c_i."""
        for k in self.c:
            for d in deltas:
                if k in d:
                    self.c[k] += d[k] / max(1, n_total_clients)
