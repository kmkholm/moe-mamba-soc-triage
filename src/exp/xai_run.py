"""XAI sweep — SHAP, LIME and permutation importance for every corpus.

    python exp/xai_run.py            # all three
    python exp/xai_run.py guide      # one

The centralised study only produced SHAP/LIME for LightGBM on GUIDE; AIT-ADS and
BOTSv3 had none. This fills the gap using the same tree baseline on all three so
the attributions are directly comparable, and uses identical rows/features/splits
to `exp/moe_run.py` (same SEED, same split code) so the explanations describe the
model whose numbers appear in the results tables.

Neural-model interpretability is covered intrinsically by the MoE router maps and
expert-utilisation heatmaps produced by `exp/moe_run.py`; SHAP's TreeExplainer
does not apply to them and KernelExplainer over a 111M-parameter model is not
tractable here.

Output: figures_xai/<dataset>/
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split

from common import evaluate as EV
from exp.moe_run import build_generic

BASE = r"D:\Drive D\Ajloun Papers\paper 57 tylor\oipfl"
FIG = os.path.join(BASE, "figures_xai")
SEED = 42
GUIDE_ROWS = 1_500_000
LGB_TREES = 400


def fit_tree(Xdf, y, tr, n_classes):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=LGB_TREES, num_leaves=127,
                           learning_rate=0.08, class_weight="balanced",
                           random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xdf.iloc[tr], y[tr])
    return m


def run(name, Xdf, y, classes):
    outdir = os.path.join(FIG, name)
    os.makedirs(outdir, exist_ok=True)
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    _, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    print(f"  fitting LightGBM ({len(tr):,} rows)...", flush=True)
    t0 = time.time()
    model = fit_tree(Xdf, y, tr, len(classes))
    print(f"  fitted in {time.time()-t0:.0f}s", flush=True)

    Xte = Xdf.iloc[te]

    print("  SHAP ...", flush=True)
    try:
        EV.shap_tree(model, Xte, outdir, f"LightGBM_{name}")
    except Exception as e:
        print(f"    shap failed: {e}", flush=True)

    print("  LIME ...", flush=True)
    try:
        EV.lime_explain(
            lambda a: model.predict_proba(pd.DataFrame(a, columns=Xdf.columns)),
            Xte.reset_index(drop=True), y[te], classes, outdir,
            f"LightGBM_{name}")
    except Exception as e:
        print(f"    lime failed: {e}", flush=True)

    print("  permutation importance ...", flush=True)
    try:
        EV.permutation_importance_fig(model, Xte, y[te], outdir,
                                      f"LightGBM_{name}")
    except Exception as e:
        print(f"    permutation failed: {e}", flush=True)

    print(f"  -> {outdir}", flush=True)


def main():
    which = [a.lower() for a in sys.argv[1:]] or ["guide", "ait", "bots"]
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()

    for w in which:
        print("=" * 70, flush=True)
        if w == "guide":
            from data.guide_full import NUM_FIELDS, load_guide_full
            print("XAI on GUIDE", flush=True)
            recs, y, _, cats = load_guide_full(nrows=GUIDE_ROWS)
            Xdf, _, _, _ = build_generic(recs, cats, NUM_FIELDS)
            run("GUIDE", Xdf, y,
                ["BenignPositive", "FalsePositive", "TruePositive"])
        elif w == "ait":
            from data.ait import CAT_FIELDS, NUM_FIELDS, load_ait
            print("XAI on AIT-ADS", flush=True)
            recs, y, _ = load_ait(verbose=False)
            Xdf, _, _, _ = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
            run("AIT-ADS", Xdf, y, ["benign", "attack"])
        elif w == "bots":
            from data.bots_rich import CAT_FIELDS, NUM_FIELDS, load_bots_rich
            print("XAI on BOTSv3", flush=True)
            recs, y, _, _ = load_bots_rich(verbose=False)
            Xdf, _, _, _ = build_generic(recs, CAT_FIELDS, NUM_FIELDS)
            run("BOTSv3", Xdf, y, ["benign", "malicious"])
        else:
            print(f"unknown dataset '{w}'", flush=True)

    print(f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
