"""Federated MoE-Mamba — ONE dataset per invocation.

    python exp/fed_run.py ait     # clients = 8 AIT-ADS scenarios
    python exp/fed_run.py guide   # clients = GUIDE organisations (top-K by volume)
    python exp/fed_run.py bots    # clients = BOTSv3 hosts (top-K by volume)

Keeps the MoE-Mamba model unchanged and federates the *training*. Clients are
the natural administrative entities already present in each corpus, so the
partition is genuinely non-IID rather than a synthetic Dirichlet split.

FedAvg is run only as the baseline. The strategies under test are the ones that
correct client drift: FedProx, FedAdam (server-side adaptive) and SCAFFOLD
(control variates).

Outputs go to NEW locations so nothing already produced is touched:
    figures_fed/<dataset>/    convergence, per-client, comparison, eval, routing
    results/fed_<dataset>.json / .csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from common import evaluate as EV
from common.moe_mamba import MoEMambaTab
from fed.fed_core import (ScaffoldState, ServerOptimizer, scaffold_feasible,
                          weighted_average)
from exp.moe_run import build_generic

BASE = r"D:\Drive D\Ajloun Papers\paper 57 tylor\oipfl"
FIG = os.path.join(BASE, "figures_fed")
RES = os.path.join(BASE, "results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROUNDS = 40          # 15 was far too few: both algorithms were still climbing
LOCAL_EPOCHS = 1     # at the final round on GUIDE (FedAvg 0.703->0.713->0.722)
BATCH, LR = 1024, 3e-3
N_EXPERTS, TOP_K, AUX_W = 8, 2, 0.01
PROX_MU = 0.01
SERVER_LR = 0.05
MAX_CLIENTS = 10
MAX_CLIENTS_BY = {"guide": 10}
MIN_CLIENT_ROWS = 2000
# Back to 20 after testing 1. Relaxing it admitted GUIDE's degenerate mega-orgs
# -- org 0 (133,081 rows, 100% TruePositive), org 2 (37,921, 100% BenignPositive),
# org 3 (30,004, 100% BenignPositive). Org 0 alone is 37% of the training data
# and single-class, so weighted averaging drags the global model toward its label
# every round and the remaining clients spend their update undoing it: val
# macro-F1 crawled 0.2581 -> 0.3853 (r10) -> 0.4163 (r20), against 0.9837 for the
# federation that excludes them. More data is not better when the extra data is
# label-degenerate.
MIN_CLIENT_PER_CLASS = 20
GUIDE_ROWS = 1_500_000

# FedProx dropped from the default sweep: 0.0000 / -0.0269 / -0.2979 across the
# three corpora, i.e. it never once helped. SCAFFOLD stays for the SGD corpora
# and is auto-skipped where the local optimiser is AdamW.
ALGOS = ["FedAdam", "FedAvg", "SCAFFOLD", "FedProx"]
ALGOS_BY = {"guide": ["FedAdam", "FedAvg", "FedProx"]}

# Local optimiser per corpus. SGD is the textbook choice for federated local
# steps and is fine for the small AIT-ADS / BOTSv3 encoders (76K / 119K params).
# It fails outright on GUIDE: ~99% of that model's 111M parameters are sparse
# embedding rows, each updated only when its token appears, so SGD at lr=3e-3
# leaves them essentially untouched -- the global model sat at macro-F1 0.2176
# (the majority-class floor) for 7 rounds. AdamW's per-parameter scaling is what
# the centralised run used to reach 0.93 in one epoch.
LOCAL_OPT = {"guide": "adamw", "ait": "sgd", "bots": "sgd"}

# Preprocessing parity with the centralised study. The centralised GUIDE result
# (MoE-Mamba 0.9773) keeps all 38 fields including Id/OrgId/IncidentId/AlertId
# and splits rows at random, so the federated arm must do the same for the
# comparison to be like-for-like. Set both True for the identifier-free
# robustness protocol instead (reported separately as GUIDE-noid).
FED_DROP_IDENTIFIERS = False
FED_GROUPED_SPLIT = False

# Ceiling on the ratio between the largest and smallest local class weight.
WEIGHT_CAP = 20.0

CKPT_EVERY = 5          # rounds between global-state checkpoints


# --------------------------------------------------------------------------
def make_clients(groups: np.ndarray, y: np.ndarray, max_clients: int):
    """Natural partition: keep the largest `max_clients` entities.

    A client must have enough of EVERY class to survive a 70/15/15 stratified
    split, not merely two distinct classes. BOTSv3 hosts are extreme here --
    some carry a single malicious event out of tens of thousands -- and one
    such host is enough to abort the whole run.
    """
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    keep, skipped = [], []
    for i in order:
        idx = np.where(groups == uniq[i])[0]
        if len(idx) < MIN_CLIENT_ROWS:
            continue
        cls_counts = np.bincount(y[idx], minlength=int(y.max()) + 1)
        present = cls_counts[cls_counts > 0]
        if len(present) < 2:
            skipped.append((str(uniq[i]), len(idx), "single-class"))
            continue
        if present.min() < MIN_CLIENT_PER_CLASS:
            skipped.append((str(uniq[i]), len(idx),
                            f"rarest class has {int(present.min())}"))
            continue
        keep.append((str(uniq[i]), idx))
        if len(keep) >= max_clients:
            break
    for cid, n, why in skipped[:10]:
        print(f"    [skip] {cid:28s} n={n:>8,}  {why}", flush=True)
    if skipped:
        print(f"    ({len(skipped)} entities excluded, {len(keep)} clients kept)",
              flush=True)
    return keep


def split_client(idx: np.ndarray, y: np.ndarray, sub: np.ndarray | None = None):
    """70/15/15 within a client.

    If `sub` is given (e.g. IncidentId), split by GROUP so no incident spans the
    client's train/test boundary. Without this, evidence rows from one incident
    land on both sides and the split leaks.
    """
    if sub is not None:
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        a, b = next(g1.split(idx, y[idx], groups=sub[idx]))
        tr, tmp = idx[a], idx[b]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
        c, d = next(g2.split(tmp, y[tmp], groups=sub[tmp]))
        return tr, tmp[c], tmp[d]

    def _split(ii, frac, strat):
        try:
            return train_test_split(ii, test_size=frac, stratify=strat,
                                    random_state=SEED)
        except ValueError:
            return train_test_split(ii, test_size=frac, random_state=SEED)

    tr, tmp = _split(idx, 0.30, y[idx])
    va, te = _split(tmp, 0.50, y[tmp])
    return tr, va, te


def loader(Xc, Xn, y, idx, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(Xc[idx]),
                                    torch.from_numpy(Xn[idx]),
                                    torch.from_numpy(y[idx])),
                      batch_size=BATCH, shuffle=shuffle, num_workers=0,
                      pin_memory=False)


def new_model(Xc, sizes, Xn, n_classes):
    torch.manual_seed(SEED)
    return MoEMambaTab(Xc.shape[1], sizes, Xn.shape[1], n_classes,
                       n_experts=N_EXPERTS, top_k=TOP_K).to(DEVICE)


@torch.no_grad()
def predict(model, Xc, Xn, idx, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        lo = model(torch.from_numpy(Xc[b]).to(DEVICE),
                   torch.from_numpy(Xn[b]).to(DEVICE))
        out.append(lo.softmax(-1).cpu().numpy())
    return np.concatenate(out)


def macro_f1(model, Xc, Xn, y, idx):
    return f1_score(y[idx], predict(model, Xc, Xn, idx).argmax(1),
                    average="macro", zero_division=0)


def local_train(model, Xc, Xn, y, idx, n_classes, algo, global_state,
                correction=None, local_opt="sgd"):
    """One client's local update. Returns (state, n_steps)."""
    dl = loader(Xc, Xn, y, idx, True)
    # Balanced weights over the classes this client actually holds. The previous
    # `cnt[cnt == 0] = 1` gave an ABSENT class the largest weight of all, so a
    # client that had never seen a class was trained hardest to predict it --
    # damaging on GUIDE, where organisations are 95-99.9% single-class.
    cnt = np.bincount(y[idx], minlength=n_classes).astype(float)
    present = cnt > 0
    w_np = np.zeros(n_classes, dtype=np.float32)
    w_np[present] = cnt[present].sum() / (present.sum() * cnt[present])
    # Cap the spread. Un-capped balanced weighting explodes on near-degenerate
    # clients: org 0 holds 133,081 rows with 4 of its rarest class, giving that
    # class a weight of ~16,600 and a gradient large enough to destroy the
    # global model (val macro-F1 frozen at 0.2240 for 5 straight rounds).
    w_np = np.minimum(w_np, w_np[present].min() * WEIGHT_CAP)
    w = torch.tensor(w_np, dtype=torch.float32, device=DEVICE)
    if local_opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    steps = 0
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for xc, xn, yb in dl:
            xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
            loss = F.cross_entropy(model(xc, xn), yb, weight=w)
            loss = loss + AUX_W * model.last_aux
            if algo == "FedProx":
                prox = 0.0
                for n_, p_ in model.named_parameters():
                    if p_.is_floating_point():
                        prox = prox + ((p_ - global_state[n_].to(DEVICE))
                                       .pow(2).sum())
                loss = loss + 0.5 * PROX_MU * prox
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if algo == "SCAFFOLD" and correction is not None:
                for n_, p_ in model.named_parameters():
                    if p_.grad is not None and n_ in correction:
                        p_.grad.add_(correction[n_].to(DEVICE))
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, steps


# --------------------------------------------------------------------------
def run_algo(algo, clients, splits, Xc, Xn, y, sizes, n_classes, va_all,
             local_opt="sgd", tag_ds="ds"):
    print(f"\n  --- {algo} ---", flush=True)
    model = new_model(Xc, sizes, Xn, n_classes)
    gstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    server = ServerOptimizer(gstate, lr=SERVER_LR) if algo == "FedAdam" else None
    scaf = (ScaffoldState(gstate, [c for c, _ in clients])
            if algo == "SCAFFOLD" else None)

    hist = []
    best, best_state = -1.0, None
    n_total = sum(len(splits[c][0]) for c, _ in clients)

    # Round-level checkpoint. The D: volume drops out roughly hourly (Windows
    # Application event 1005 / 0xC0000005) and a 40-round algorithm takes about
    # as long, so without this an algorithm can restart forever and never finish.
    ckpt_dir = os.path.join(RES, "_fed_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"{tag_ds}_{algo}.pt")
    start_round = 1
    if os.path.exists(ckpt):
        try:
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            gstate = blob["gstate"]
            hist = blob["hist"]
            best, best_state = blob["best"], blob["best_state"]
            start_round = blob["round"] + 1
            if server is not None and blob.get("server") is not None:
                server.m, server.v, server.t = blob["server"]
            print(f"      [ckpt] resuming {algo} from round {start_round}",
                  flush=True)
        except Exception as e:
            print(f"      [ckpt] unreadable ({e}), starting fresh", flush=True)

    for rnd in range(start_round, ROUNDS + 1):
        # Streaming aggregation: accumulate the weighted sum as each client
        # finishes instead of holding all client states at once. With the
        # 111M-parameter GUIDE model and 10 clients that is the difference
        # between ~0.9 GB and ~4.4 GB of host RAM.
        acc, deltas = None, []
        for cid, _ in clients:
            i_tr = splits[cid][0]
            model.load_state_dict(gstate)
            corr = scaf.correction(cid) if scaf else None
            st, steps = local_train(model, Xc, Xn, y, i_tr, n_classes, algo,
                                    gstate, corr, local_opt)
            if scaf:
                deltas.append(scaf.update_client(cid, gstate, st, steps, LR))
            frac = len(i_tr) / n_total
            if acc is None:
                acc = {k: (v.to(torch.float32) * frac if v.is_floating_point()
                           else v.clone()) for k, v in st.items()}
            else:
                for k, v in st.items():
                    if v.is_floating_point():
                        acc[k] += v.to(torch.float32) * frac
            del st

        agg = {k: v.to(gstate[k].dtype) for k, v in acc.items()}
        del acc
        if server is not None:
            gstate = server.step(gstate, agg)
        else:
            gstate = agg
        if scaf:
            scaf.update_server(deltas, len(clients))

        model.load_state_dict(gstate)
        f_va = macro_f1(model, Xc, Xn, y, va_all)
        hist.append({"round": rnd, "val_f1": float(f_va)})
        print(f"      round {rnd:2d}/{ROUNDS}  val_macroF1={f_va:.4f}", flush=True)
        if f_va > best:
            best, best_state = f_va, {k: v.clone() for k, v in gstate.items()}

        if rnd % CKPT_EVERY == 0 or rnd == ROUNDS:
            try:
                torch.save({"gstate": gstate, "hist": hist, "best": best,
                            "best_state": best_state, "round": rnd,
                            "server": ((server.m, server.v, server.t)
                                       if server is not None else None)},
                           ckpt)
            except OSError as e:
                print(f"      [ckpt] save failed: {e}", flush=True)

    model.load_state_dict(best_state)
    try:
        os.remove(ckpt)          # algorithm finished; checkpoint no longer needed
    except OSError:
        pass
    return model, hist, best


def plot_convergence(all_hist, outdir, tag):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for algo, h in all_hist.items():
        style = "--" if algo == "FedAvg" else "-"
        ax.plot([r["round"] for r in h], [r["val_f1"] for r in h], style,
                marker="o", ms=3, label=algo)
    ax.set_xlabel("communication round")
    ax.set_ylabel("global validation macro-F$_1$")
    ax.set_title(f"{tag} — federated convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fed_convergence_{tag}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_client(per_client, outdir, tag):
    df = pd.DataFrame(per_client)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * df["client"].nunique()), 4.2))
    sns.barplot(data=df, x="client", y="macro_f1", hue="algo", ax=ax)
    ax.set_xlabel("client")
    ax.set_ylabel("test macro-F$_1$")
    ax.set_title(f"{tag} — per-client performance")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fed_per_client_{tag}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_client_distribution(clients, y, n_classes, outdir, tag):
    rows = []
    for cid, idx in clients:
        cnt = np.bincount(y[idx], minlength=n_classes)
        for c in range(n_classes):
            rows.append({"client": cid, "class": f"class {c}",
                         "fraction": cnt[c] / max(1, cnt.sum()),
                         "n": int(cnt.sum())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(clients)), 4.2))
    piv = df.pivot(index="client", columns="class", values="fraction")
    piv.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("class fraction")
    ax.set_title(f"{tag} — client label distribution (non-IID check)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fed_client_dist_{tag}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "ait").lower()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    sub_groups = None      # optional within-client grouping key (e.g. IncidentId)

    if which == "ait":
        from data.ait import CAT_FIELDS, NUM_FIELDS, load_ait
        name, classes = "AIT-ADS", ["benign", "attack"]
        recs, y, groups = load_ait(verbose=False)
        Xdf, Xc, Xn, sizes = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
        unit = "scenario"
    elif which == "bots":
        from data.bots_rich import CAT_FIELDS, NUM_FIELDS, load_bots_rich
        name, classes = "BOTSv3", ["benign", "malicious"]
        recs, y, groups, _ = load_bots_rich(verbose=False)
        Xdf, Xc, Xn, sizes = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
        unit = "host"
    elif which == "guide":
        from data.guide_full import NUM_FIELDS, load_guide_full
        name, classes = "GUIDE", ["BenignPositive", "FalsePositive", "TruePositive"]
        recs, y, groups, cats = load_guide_full(nrows=GUIDE_ROWS)
        org = np.array([g.split("|")[0] for g in groups])
        sub = np.array([g.split("|")[1] for g in groups])       # IncidentId
        groups = org                                            # clients = orgs
        if FED_DROP_IDENTIFIERS:
            # OrgId is constant inside a client, so it carries no legitimate
            # within-client signal -- it only lets the model memorise
            # "org -> its dominant grade". With 5 near-single-class orgs that
            # lookup alone scores ~0.988 accuracy, which is what the leaky run
            # actually learned. Id/AlertId/IncidentId leak the same way.
            drop = {"Id", "OrgId", "IncidentId", "AlertId"}
            cats = [c for c in cats if c not in drop]
            name = "GUIDE-noid"
            print(f"  identifier fields dropped: {sorted(drop)}", flush=True)
        Xdf, Xc, Xn, sizes = build_generic(recs, cats, NUM_FIELDS)
        unit = "organisation"
        if FED_GROUPED_SPLIT:
            sub_groups = sub
            print("  within-client splits grouped by IncidentId", flush=True)
    else:
        raise SystemExit(f"unknown dataset '{which}' (ait|bots|guide)")

    n_classes = len(classes)
    print("=" * 70)
    print(f"Federated MoE-Mamba on {name} (clients = {unit}s)")
    print("=" * 70, flush=True)

    clients = make_clients(groups, y, MAX_CLIENTS_BY.get(which, MAX_CLIENTS))
    splits = {}
    tr_all, va_all, te_all = [], [], []
    for cid, idx in clients:
        tr, va, te = split_client(idx, y, sub_groups)
        splits[cid] = (tr, va, te)
        tr_all.append(tr); va_all.append(va); te_all.append(te)
    va_all = np.concatenate(va_all)
    te_all = np.concatenate(te_all)
    print(f"  {len(clients)} clients, train={sum(len(splits[c][0]) for c,_ in clients):,} "
          f"val={len(va_all):,} test={len(te_all):,}", flush=True)
    for cid, idx in clients:
        cnt = np.bincount(y[idx], minlength=n_classes)
        print(f"    {cid:28s} n={len(idx):>8,}  dist={ (cnt/cnt.sum()).round(3).tolist() }",
              flush=True)

    outdir = os.path.join(FIG, name)
    os.makedirs(outdir, exist_ok=True)
    plot_client_distribution(clients, y, n_classes, outdir, name)

    local_opt = LOCAL_OPT.get(which, "sgd")
    print(f"  local optimiser: {local_opt}", flush=True)

    n_params = sum(p.numel() for p in new_model(Xc, sizes, Xn, n_classes).parameters())
    algos = list(ALGOS_BY.get(which, ALGOS))
    if "SCAFFOLD" in algos and not scaffold_feasible(n_params, len(clients)):
        print(f"  SCAFFOLD skipped: {n_params:,} params x {len(clients)+1} copies "
              f"exceeds the memory budget", flush=True)
        algos.remove("SCAFFOLD")
    if "SCAFFOLD" in algos and local_opt != "sgd":
        # NOT APPLICABLE, established empirically. The control-variate update
        # divides the parameter change by (n_steps * lr), which assumes an
        # SGD-sized local step; under AdamW the step comes from the adaptive
        # preconditioner and bears no relation to lr, so the variates are
        # mis-scaled by orders of magnitude. Run once anyway on GUIDE: macro-F1
        # 0.2715 with MCC -0.0986, i.e. anti-correlated with the labels.
        # And SGD is not an option here either -- it leaves the 111M-parameter
        # embedding model frozen at the majority-class floor (0.2176). There is
        # no configuration where both SCAFFOLD's assumption and the model's
        # optimisation requirement hold, so the corpus is reported as n/a.
        print(f"  SCAFFOLD skipped: not applicable under {local_opt} "
              f"(control variates assume SGD-sized steps)", flush=True)
        algos.remove("SCAFFOLD")

    # Resume: the D: volume has dropped out mid-run three times in this project
    # (Application event 1005, "Windows cannot access the file ... the disk is
    # missing"), each time killing python with no traceback. Reload whatever
    # finished and skip those algorithms rather than repeating them.
    res_path = os.path.join(RES, f"fed_{name}.json")
    prev = {}
    if os.path.exists(res_path):
        try:
            with open(res_path, encoding="utf-8") as f:
                prev = json.load(f).get(name, {})
        except Exception:
            prev = {}
    done = [a for a in algos if isinstance(prev.get(a), dict)
            and "f1_macro" in prev[a]]
    if done:
        print(f"  resuming — already on disk: {done}", flush=True)
        algos = [a for a in algos if a not in done]

    metrics = {name: {"config": {
        "clients": len(clients), "client_unit": unit, "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS, "batch": BATCH, "lr": LR,
        "prox_mu": PROX_MU, "server_lr": SERVER_LR, "params": int(n_params),
        "local_optimiser": local_opt,
        "client_sizes": {cid: int(len(idx)) for cid, idx in clients}}}}
    rows, all_hist, per_client = [], {}, []

    # carry completed algorithms forward so the merged file stays complete
    for a in done:
        metrics[name][a] = prev[a]
        all_hist[a] = prev[a].get("rounds_history", [])
        rows.append({"dataset": name, "algo": a,
                     "macro_f1": prev[a]["f1_macro"],
                     "accuracy": prev[a]["accuracy"], "mcc": prev[a]["mcc"],
                     "best_val": prev[a].get("best_val_macro_f1", float("nan"))})
    per_client.extend(prev.get("_per_client", []))

    # Resume: three runs so far have been killed mid-experiment by an
    # intermittent native access violation (0xC0000005) with no Python
    # traceback. Reload whatever finished and skip those algorithms.
    res_path = os.path.join(RES, f"fed_{name}.json")
    if os.path.exists(res_path):
        try:
            with open(res_path, encoding="utf-8") as f:
                prev = json.load(f).get(name, {})
        except Exception:
            prev = {}
        for algo in list(algos):
            v = prev.get(algo)
            if isinstance(v, dict) and "f1_macro" in v:
                metrics[name][algo] = v
                all_hist[algo] = v.get("rounds_history", [])
                rows.append({"dataset": name, "algo": algo,
                             "macro_f1": v["f1_macro"], "accuracy": v["accuracy"],
                             "mcc": v["mcc"],
                             "best_val": v.get("best_val_macro_f1", 0.0)})
                algos.remove(algo)
                print(f"  [resume] {algo} already complete "
                      f"(macro-F1 {v['f1_macro']:.4f}) — skipping", flush=True)
        per_client = [p for p in prev.get("per_client", [])
                      if p.get("algo") in metrics[name]]

    for algo in algos:
        model, hist, best_val = run_algo(algo, clients, splits, Xc, Xn, y,
                                         sizes, n_classes, va_all, local_opt,
                                         name)
        all_hist[algo] = hist

        proba = predict(model, Xc, Xn, te_all)
        m = EV.full_metrics(y[te_all], proba.argmax(1), proba, classes)
        m["rounds_history"] = hist
        m["best_val_macro_f1"] = float(best_val)
        m["expert_utilisation"] = model.expert_utilisation().tolist()
        m["active_params_per_token"] = int(model.active_params())
        metrics[name][algo] = m
        rows.append({"dataset": name, "algo": algo, "macro_f1": m["f1_macro"],
                     "accuracy": m["accuracy"], "mcc": m["mcc"],
                     "best_val": float(best_val)})
        print(f"    {algo:9s} TEST macroF1={m['f1_macro']:.4f} "
              f"acc={m['accuracy']:.4f} mcc={m['mcc']:.4f}", flush=True)

        for cid, _ in clients:
            te = splits[cid][2]
            per_client.append({"client": cid, "algo": algo,
                               "macro_f1": macro_f1(model, Xc, Xn, y, te)})

        tag = f"{algo}_{name}"
        EV.plot_confusion(y[te_all], proba.argmax(1), classes, outdir, tag)
        EV.plot_roc(y[te_all], proba, classes, outdir, tag)
        EV.plot_pr(y[te_all], proba, classes, outdir, tag)
        EV.plot_calibration(y[te_all], proba, outdir, tag)

        metrics[name]["_per_client"] = per_client
        with open(os.path.join(RES, f"fed_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"fed_{name}_summary.csv"),
                                  index=False)
        print(f"    [saved] results/fed_{name}.json ({algo} done)", flush=True)

    plot_convergence(all_hist, outdir, name)
    plot_per_client(per_client, outdir, name)
    metrics[name]["per_client"] = per_client

    with open(os.path.join(RES, f"fed_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"figures -> {outdir}")
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
