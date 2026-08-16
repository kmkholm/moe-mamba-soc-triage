"""MoE-Mamba experiment — ONE dataset per invocation.

    python exp/moe_run.py guide     # GUIDE, full 38-col set (identifiers in)
    python exp/moe_run.py ait       # AIT-ADS, 12 fields
    python exp/moe_run.py bots      # BOTSv3, 26 rich behavioural fields

Uses exactly the preprocessing that produced the high-accuracy runs, and
compares three models on identical rows / features / splits:

    MoE-Mamba   (new)   BiMamba + sparse top-k MoE-FFN per layer
    Mamba       (dense) the previous architecture
    LightGBM            strongest tree baseline

Outputs go to NEW locations so nothing already produced is touched:
    figures_moe/<dataset>/   all plots (+ expert-utilisation & routing maps)
    results/moe_<dataset>.json / .csv
"""
from __future__ import annotations

import json
import os
import sys
import time

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
from common.mamba_tab import MambaTab
from common.moe_mamba import MoEMambaTab

BASE = r"D:\Drive D\Ajloun Papers\paper 57 tylor\oipfl"
FIG = os.path.join(BASE, "figures_moe")
RES = os.path.join(BASE, "results")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH, EPOCHS, LR = 1024, 12, 3e-3
N_EXPERTS, TOP_K, AUX_W = 8, 2, 0.01
GUIDE_ROWS = 1_500_000
LGB_TREES = 400

# Tree baselines are NOT retrained here. eval_all.py already fitted LightGBM and
# RandomForest on byte-identical splits (same SEED, same split code) — GUIDE-full
# stratified LightGBM matches to 16 decimals in both scripts — so the baseline
# column is taken from results/eval_all.json instead of being recomputed.
# Set to True only if a dataset has no matched baseline on record.
TRAIN_BASELINE = False


# --------------------------------------------------------------------------
def build_generic(recs, cat_fields, num_fields, max_card=200_000):
    df = pd.DataFrame(recs)
    for c in cat_fields + num_fields:
        if c not in df.columns:
            df[c] = np.nan
    codes_list, sizes = [], []
    for c in cat_fields:
        codes = pd.factorize(df[c].astype("string").fillna("__NA__"))[0].astype(np.int64)
        if codes.max() + 1 > max_card:
            codes = codes % max_card
        codes_list.append(codes); sizes.append(int(codes.max()) + 2)
    Xc = np.stack(codes_list, axis=1)
    Xn_raw = np.stack([pd.to_numeric(df[c], errors="coerce").fillna(-1).to_numpy()
                       for c in num_fields], axis=1).astype(np.float32)
    Xdf = pd.DataFrame(np.concatenate([Xc.astype(np.float32), Xn_raw], axis=1),
                       columns=cat_fields + num_fields)
    mu, sd = Xn_raw.mean(0, keepdims=True), Xn_raw.std(0, keepdims=True) + 1e-6
    return Xdf, Xc, (Xn_raw - mu) / sd, sizes


def train_nn(model, Xc, Xn, y, tr, va, n_classes, is_moe: bool):
    torch.manual_seed(SEED); np.random.seed(SEED)

    def mk(ii, sh):
        # pin_memory=False on purpose: pinned-host-buffer allocation is the one
        # path implicated in the intermittent 0xC0000005 access violations
        # inside torch_cpu.dll that killed two overnight runs (GUIDE, BOTSv3).
        # Costs a few % throughput; buys a run that survives to completion.
        return DataLoader(TensorDataset(torch.from_numpy(Xc[ii]),
                                        torch.from_numpy(Xn[ii]),
                                        torch.from_numpy(y[ii])),
                          batch_size=BATCH, shuffle=sh, num_workers=0,
                          pin_memory=False)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    cnt = np.bincount(y[tr], minlength=n_classes).astype(float); cnt[cnt == 0] = 1
    w = torch.tensor(cnt.sum() / (n_classes * cnt), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                              total_steps=EPOCHS * len(dl_tr))
    hist, best, state = [], -1.0, None
    for e in range(EPOCHS):
        model.train(); tot, aux_tot, P, Y = 0.0, 0.0, [], []
        for xc, xn, yb in dl_tr:
            xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
            lo = model(xc, xn)
            loss = F.cross_entropy(lo, yb, weight=w)
            if is_moe:
                aux = model.last_aux
                loss = loss + AUX_W * aux
                aux_tot += float(aux.detach())
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tot += float(loss.detach()) * len(yb)
            P.append(lo.argmax(-1).detach().cpu().numpy()); Y.append(yb.cpu().numpy())
        ftr = f1_score(np.concatenate(Y), np.concatenate(P), average="macro",
                       zero_division=0)
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for xc, xn, yb in dl_va:
                lo = model(xc.to(DEVICE), xn.to(DEVICE))
                P.append(lo.argmax(-1).cpu().numpy()); Y.append(yb.numpy())
        fva = f1_score(np.concatenate(Y), np.concatenate(P), average="macro",
                       zero_division=0)
        row = {"epoch": e + 1, "train_loss": tot / max(1, len(tr)),
               "train_f1": float(ftr), "val_f1": float(fva)}
        if is_moe:
            row["aux_loss"] = aux_tot / max(1, len(dl_tr))
        hist.append(row)
        print(f"      ep{e+1}/{EPOCHS} loss={row['train_loss']:.4f} "
              f"train={ftr:.4f} val={fva:.4f}"
              + (f" aux={row['aux_loss']:.4f}" if is_moe else ""), flush=True)
        if fva > best:
            best = fva
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, hist


@torch.no_grad()
def nn_proba(model, Xc, Xn, idx, bs=4096):
    model.eval(); out = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        lo = model(torch.from_numpy(Xc[b]).to(DEVICE),
                   torch.from_numpy(Xn[b]).to(DEVICE))
        out.append(lo.softmax(-1).cpu().numpy())
    return np.concatenate(out)


def plot_experts(util: np.ndarray, outdir: str, tag: str):
    fig, ax = plt.subplots(figsize=(7, 3 + 0.4 * util.shape[0]))
    sns.heatmap(util, annot=True, fmt=".3f", cmap="viridis", ax=ax,
                xticklabels=[f"E{i}" for i in range(util.shape[1])],
                yticklabels=[f"layer {i}" for i in range(util.shape[0])])
    ax.set_title(f"{tag} — expert utilisation (fraction of tokens)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"expert_util_{tag}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def plot_routing_by_field(model, Xc, Xn, idx, field_names, outdir, tag, cap=8000):
    """Mean router probability per (field, expert) — which fields share experts."""
    sub = idx[:cap]
    with torch.no_grad():
        _, routing = model(torch.from_numpy(Xc[sub]).to(DEVICE),
                           torch.from_numpy(Xn[sub]).to(DEVICE),
                           return_routing=True)
    for li, r in enumerate(routing):
        m = r.mean(0).numpy()                       # [L, E]
        fig, ax = plt.subplots(figsize=(6, max(4, 0.22 * len(field_names))))
        sns.heatmap(m, cmap="magma", ax=ax,
                    xticklabels=[f"E{i}" for i in range(m.shape[1])],
                    yticklabels=field_names)
        ax.set_title(f"{tag} — layer {li} routing (mean router prob per field)")
        ax.tick_params(axis="y", labelsize=6)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"routing_L{li}_{tag}.{ext}"), dpi=200,
                        bbox_inches="tight")
        plt.close(fig)


MODELS = ("LightGBM", "Mamba", "MoE-Mamba") if TRAIN_BASELINE else ("Mamba", "MoE-Mamba")


def _row(name, mdl_name, sp, m):
    return {"dataset": name, "model": mdl_name, "split": sp,
            "macro_f1": m["f1_macro"], "accuracy": m["accuracy"], "mcc": m["mcc"]}


def save_partial(name, metrics, rows):
    """Dump after every split so a crash costs at most one split, not the run."""
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"moe_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"moe_{name}_summary.csv"),
                              index=False)


def load_partial(name):
    """Resume: returns (metrics, rows) from a previous interrupted run."""
    p = os.path.join(RES, f"moe_{name}.json")
    if not os.path.exists(p):
        return {name: {}}, []
    try:
        with open(p, encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        return {name: {}}, []
    metrics.setdefault(name, {})
    rows = [_row(name, mdl, sp, m)
            for mdl, spd in metrics[name].items() for sp, m in spd.items()]
    done = sorted({sp for spd in metrics[name].values() for sp in spd})
    if done:
        print(f"  resuming — splits already on disk: {done}", flush=True)
    return metrics, rows


# --------------------------------------------------------------------------
def run(name, Xdf, Xc, Xn, y, sizes, groups, group_label, class_names,
        max_group_splits=2):
    outdir = os.path.join(FIG, name)
    os.makedirs(outdir, exist_ok=True)
    n_classes = len(class_names)
    idx = np.arange(len(y))
    metrics, rows = load_partial(name)

    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    splits = {"stratified": (tr, va, te)}
    if groups is not None:
        if group_label == "incident":
            g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
            tr_g, tmp_g = next(g1.split(Xdf, y, groups=groups))
            g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
            r1, r2 = next(g2.split(Xdf.iloc[tmp_g], y[tmp_g], groups=groups[tmp_g]))
            splits["incident_grouped"] = (tr_g, tmp_g[r1], tmp_g[r2])
        else:
            # eval_all gave RandomForest / LightGBM / Mamba the identical score
            # 0.4840436023194961 on this host — every model collapsed to the
            # majority class, so the split carries no signal. ~2h to reproduce
            # a known-degenerate number; the negative cross-host transfer result
            # is already on record in results/eval_all.json.
            degenerate = {"splunkhwf.froth.ly"}
            pos = {g: int(y[groups == g].sum()) for g in np.unique(groups)
                   if g not in degenerate}
            for g, c in sorted(pos.items(), key=lambda kv: -kv[1])[:max_group_splits]:
                te_g = np.where(groups == g)[0]; rest = np.where(groups != g)[0]
                if y[te_g].sum() < 5 or len(np.unique(y[te_g])) < 2:
                    continue
                tr_g, va_g = train_test_split(rest, test_size=0.15,
                                              stratify=y[rest], random_state=SEED)
                splits[f"{group_label}_{g}"] = (tr_g, va_g, te_g)

    for sp, (i_tr, i_va, i_te) in splits.items():
        if all(sp in metrics[name].get(mn, {}) for mn in MODELS):
            print(f"\n  [{name} / {sp}] already complete — skipping", flush=True)
            continue
        print(f"\n  [{name} / {sp}] train={len(i_tr):,} val={len(i_va):,} "
              f"test={len(i_te):,}", flush=True)

        # ---- LightGBM baseline (off by default — see TRAIN_BASELINE) ----
        lg = None
        if TRAIN_BASELINE:
            import lightgbm as lgb
            t0 = time.time()
            lg = lgb.LGBMClassifier(n_estimators=LGB_TREES, num_leaves=127,
                                    learning_rate=0.08, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1, verbose=-1)
            lg.fit(Xdf.iloc[i_tr], y[i_tr])
            print(f"    LightGBM fitted ({time.time()-t0:.0f}s)", flush=True)

        # ---- dense Mamba ----
        print("    Mamba (dense):", flush=True)
        dense = MambaTab(Xc.shape[1], sizes, Xn.shape[1], n_classes).to(DEVICE)
        dense, h_dense = train_nn(dense, Xc, Xn, y, i_tr, i_va, n_classes, False)

        # ---- MoE-Mamba ----
        print(f"    MoE-Mamba (E={N_EXPERTS}, k={TOP_K}):", flush=True)
        moe = MoEMambaTab(Xc.shape[1], sizes, Xn.shape[1], n_classes,
                          n_experts=N_EXPERTS, top_k=TOP_K).to(DEVICE)
        moe, h_moe = train_nn(moe, Xc, Xn, y, i_tr, i_va, n_classes, True)

        todo = [("Mamba", dense, h_dense), ("MoE-Mamba", moe, h_moe)]
        if lg is not None:
            todo.insert(0, ("LightGBM", lg, None))
        for mdl_name, mdl, hist in todo:
            proba = (lg.predict_proba(Xdf.iloc[i_te]) if mdl_name == "LightGBM"
                     else nn_proba(mdl, Xc, Xn, i_te))
            pred = proba.argmax(1)
            m = EV.full_metrics(y[i_te], pred, proba, class_names)
            if hist:
                m["training_history"] = hist
                m["params"] = int(sum(p.numel() for p in mdl.parameters()))
            if mdl_name == "MoE-Mamba":
                m["active_params_per_token"] = int(mdl.active_params())
                m["expert_utilisation"] = mdl.expert_utilisation().tolist()
            metrics[name].setdefault(mdl_name, {})[sp] = m
            rows.append(_row(name, mdl_name, sp, m))
            print(f"    {mdl_name:11s} macroF1={m['f1_macro']:.4f} "
                  f"acc={m['accuracy']:.4f} mcc={m['mcc']:.4f}", flush=True)

            tag = f"{mdl_name}_{sp}"
            EV.plot_confusion(y[i_te], pred, class_names, outdir, tag)
            EV.plot_roc(y[i_te], proba, class_names, outdir, tag)
            EV.plot_pr(y[i_te], proba, class_names, outdir, tag)
            EV.plot_calibration(y[i_te], proba, outdir, tag)
            if hist:
                EV.plot_training_curve(hist, outdir, tag)
            if n_classes == 2:
                m["threshold_sweep"] = EV.plot_threshold_sweep(
                    y[i_te], proba[:, 1], outdir, tag)

        # MoE-specific interpretability
        plot_experts(moe.expert_utilisation(), outdir, f"MoE-Mamba_{sp}")
        try:
            plot_routing_by_field(moe, Xc, Xn, i_te, list(Xdf.columns), outdir,
                                  f"MoE-Mamba_{sp}")
        except Exception as e:
            print(f"      routing map failed: {e}", flush=True)

        if sp == "stratified" and lg is not None:
            print("    [XAI] SHAP / LIME / dt ...", flush=True)
            try:
                sh = EV.shap_tree(lg, Xdf.iloc[i_te], outdir, f"LightGBM_{sp}")
                if sh:
                    metrics[name]["LightGBM"][sp].update(sh)
            except Exception as e:
                print(f"      shap failed: {e}", flush=True)
            try:
                EV.lime_explain(lambda a: lg.predict_proba(
                    pd.DataFrame(a, columns=Xdf.columns)),
                    Xdf.iloc[i_te].reset_index(drop=True), y[i_te],
                    class_names, outdir, f"LightGBM_{sp}")
            except Exception as e:
                print(f"      lime failed: {e}", flush=True)

        del dense, moe, lg
        torch.cuda.empty_cache()
        save_partial(name, metrics, rows)
        print(f"    [saved] results/moe_{name}.json ({sp} done)", flush=True)

    EV.plot_model_comparison(rows, outdir, name)
    save_partial(name, metrics, rows)
    print("\n" + "=" * 70)
    print(pd.DataFrame(rows).pivot_table(index="split", columns="model",
                                         values="macro_f1").round(4).to_string())
    print(f"figures -> {outdir}")
    print(f"metrics -> {RES}\\moe_{name}.json")


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "guide").lower()
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()

    if which == "guide":
        from data.guide_full import NUM_FIELDS, load_guide_full
        print("=" * 70 + "\nMoE-Mamba on GUIDE (full 38-col, identifiers in)\n"
              + "=" * 70, flush=True)
        recs, y, groups, cats = load_guide_full(nrows=GUIDE_ROWS)
        Xdf, Xc, Xn, sizes = build_generic(recs, cats, NUM_FIELDS)
        run("GUIDE", Xdf, Xc, Xn, y, sizes, groups, "incident",
            ["BenignPositive", "FalsePositive", "TruePositive"])

    elif which == "ait":
        from data.ait import CAT_FIELDS, NUM_FIELDS, load_ait
        print("=" * 70 + "\nMoE-Mamba on AIT-ADS\n" + "=" * 70, flush=True)
        recs, y, scn = load_ait(verbose=False)
        Xdf, Xc, Xn, sizes = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
        run("AIT-ADS", Xdf, Xc, Xn, y, sizes, scn, "scenario",
            ["benign", "attack"])

    elif which == "bots":
        from data.bots_rich import CAT_FIELDS, NUM_FIELDS, load_bots_rich
        print("=" * 70 + "\nMoE-Mamba on BOTSv3 (26 rich fields)\n" + "=" * 70,
              flush=True)
        recs, y, host, st = load_bots_rich(verbose=False)
        Xdf, Xc, Xn, sizes = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
        # one held-out host only (the second-ranked host is degenerate — see
        # `degenerate` in run()); keeps BOTSv3 to 2 splits / ~4h
        run("BOTSv3", Xdf, Xc, Xn, y, sizes, host, "host",
            ["benign", "malicious"], max_group_splits=1)
    else:
        raise SystemExit(f"unknown dataset '{which}' (guide|ait|bots)")

    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
