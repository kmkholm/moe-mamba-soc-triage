"""Evaluation suite: full metric set + publication figures + XAI.

Produces, for any (dataset, model) pair:
  metrics  - accuracy, balanced accuracy, macro/micro/weighted P/R/F1, per-class
             P/R/F1/support, MCC, Cohen's kappa, specificity, FPR, ROC-AUC,
             PR-AUC (average precision), log loss, Brier score
  figures  - confusion matrix (counts + row-normalised), ROC (per class +
             micro/macro), precision-recall, calibration, threshold sweep,
             score distribution, training curves
  XAI      - SHAP (beeswarm / bar / dependence), LIME local explanations,
             permutation importance, and for Mamba the intrinsic dt
             selection-gate attribution

Everything is written to <FIG_DIR>/<dataset>/ as PNG + PDF, metrics to JSON+CSV.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]


def _save(fig, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------
def full_metrics(y_true, y_pred, y_proba=None, class_names: Sequence[str] = None
                 ) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_cls = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_cls))
    if class_names is None:
        class_names = [f"class_{i}" for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m: Dict = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
    }
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, labels=labels, zero_division=0)
        m[f"precision_{avg}"] = float(p)
        m[f"recall_{avg}"] = float(r)
        m[f"f1_{avg}"] = float(f)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 zero_division=0)
    per = {}
    for i, name in enumerate(class_names[:n_cls]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per[name] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
    m["per_class"] = per

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        try:
            if n_cls == 2:
                m["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                m["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
                m["brier"] = float(brier_score_loss(y_true, y_proba[:, 1]))
            else:
                m["roc_auc_ovr_macro"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"))
                m["roc_auc_ovr_weighted"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"))
                Y = np.eye(n_cls)[y_true]
                m["pr_auc_macro"] = float(average_precision_score(Y, y_proba,
                                                                 average="macro"))
            m["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
        except Exception as e:  # degenerate folds (single class present)
            m["auc_error"] = str(e)
    m["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names[:n_cls]),
        digits=4, zero_division=0, output_dict=True)
    return m


# --------------------------------------------------------------------------
# FIGURES
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, class_names, outdir, tag):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cmn = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title(f"{tag} — confusion (counts)")
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title(f"{tag} — confusion (row-normalised)")
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
    _save(fig, outdir, f"confusion_{tag}")


def plot_roc(y_true, y_proba, class_names, outdir, tag):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    n = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if n == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2,
                label=f"AUC = {roc_auc_score(y_true, y_proba[:,1]):.4f}")
    else:
        Y = np.eye(n)[y_true]
        for i, cn in enumerate(class_names[:n]):
            if Y[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, lw=1.8, color=PALETTE[i % len(PALETTE)],
                    label=f"{cn} (AUC={roc_auc_score(Y[:,i], y_proba[:,i]):.4f})")
        fpr, tpr, _ = roc_curve(Y.ravel(), y_proba.ravel())
        ax.plot(fpr, tpr, "--", color="grey", lw=1.5,
                label=f"micro (AUC={roc_auc_score(Y.ravel(), y_proba.ravel()):.4f})")
    ax.plot([0, 1], [0, 1], ":", color="black", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{tag} — ROC")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, outdir, f"roc_{tag}")


def plot_pr(y_true, y_proba, class_names, outdir, tag):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    n = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if n == 2:
        pr, rc, _ = precision_recall_curve(y_true, y_proba[:, 1])
        ax.plot(rc, pr, color=PALETTE[1], lw=2,
                label=f"AP = {average_precision_score(y_true, y_proba[:,1]):.4f}")
        ax.axhline(y_true.mean(), ls=":", color="black", lw=1,
                   label=f"baseline = {y_true.mean():.4f}")
    else:
        Y = np.eye(n)[y_true]
        for i, cn in enumerate(class_names[:n]):
            if Y[:, i].sum() == 0:
                continue
            pr, rc, _ = precision_recall_curve(Y[:, i], y_proba[:, i])
            ax.plot(rc, pr, lw=1.8, color=PALETTE[i % len(PALETTE)],
                    label=f"{cn} (AP={average_precision_score(Y[:,i], y_proba[:,i]):.4f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{tag} — precision-recall")
    ax.legend(loc="best", fontsize=8)
    _save(fig, outdir, f"pr_{tag}")


def plot_calibration(y_true, y_proba, outdir, tag, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim > 1:
        pos = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(1)
        yb = y_true if y_proba.shape[1] == 2 else (y_true == y_proba.argmax(1)).astype(int)
    else:
        pos, yb = y_proba, y_true
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    try:
        ft, mp = calibration_curve(yb, pos, n_bins=n_bins, strategy="quantile")
        ax[0].plot(mp, ft, "o-", color=PALETTE[0], label="model")
    except Exception:
        pass
    ax[0].plot([0, 1], [0, 1], ":", color="black", label="perfect")
    ax[0].set_xlabel("mean predicted probability"); ax[0].set_ylabel("observed frequency")
    ax[0].set_title(f"{tag} — calibration"); ax[0].legend(fontsize=8)
    ax[1].hist(pos, bins=40, color=PALETTE[2], alpha=0.85)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("count (log)")
    ax[1].set_title(f"{tag} — score distribution")
    _save(fig, outdir, f"calibration_{tag}")


def plot_threshold_sweep(y_true, y_score, outdir, tag):
    """Binary only: precision / recall / F1 vs decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    ths = np.linspace(0.01, 0.99, 99)
    P, R, F = [], [], []
    for t in ths:
        yp = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, yp, average="binary",
                                                     zero_division=0)
        P.append(p); R.append(r); F.append(f)
    best = int(np.argmax(F))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ths, P, label="precision", color=PALETTE[0])
    ax.plot(ths, R, label="recall", color=PALETTE[1])
    ax.plot(ths, F, label="F1", color=PALETTE[2], lw=2)
    ax.axvline(ths[best], ls="--", color="grey",
               label=f"best F1={F[best]:.4f} @ t={ths[best]:.2f}")
    ax.axvline(0.5, ls=":", color="black", lw=1, label="default t=0.50")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title(f"{tag} — threshold sweep"); ax.legend(fontsize=8)
    _save(fig, outdir, f"threshold_{tag}")
    return {"best_threshold": float(ths[best]), "best_f1": float(F[best]),
            "f1_at_0.5": float(F[np.argmin(np.abs(ths - 0.5))])}


def plot_training_curve(history: List[Dict], outdir, tag):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if "train_loss" in df:
        ax[0].plot(df["epoch"], df["train_loss"], "o-", color=PALETTE[3])
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss")
        ax[0].set_title(f"{tag} — loss")
    for col, c, lab in [("train_f1", PALETTE[0], "train"),
                        ("val_f1", PALETTE[1], "val")]:
        if col in df:
            ax[1].plot(df["epoch"], df[col], "o-", color=c, label=lab)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-F1")
    ax[1].set_title(f"{tag} — macro-F1"); ax[1].legend(fontsize=8)
    _save(fig, outdir, f"training_{tag}")


def plot_model_comparison(rows: List[Dict], outdir, tag):
    """rows: [{'model':..,'split':..,'macro_f1':..}]"""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * df["split"].nunique() * 1.5), 4.2))
    sns.barplot(data=df, x="split", y="macro_f1", hue="model", ax=ax,
                palette=PALETTE[:df["model"].nunique()])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7, padding=1)
    ax.set_ylim(0, 1.05); ax.set_ylabel("macro-F1"); ax.set_xlabel("")
    ax.set_title(f"{tag} — model comparison")
    ax.legend(fontsize=8, title=None)
    plt.xticks(rotation=15, ha="right")
    _save(fig, outdir, f"comparison_{tag}")


# --------------------------------------------------------------------------
# XAI
# --------------------------------------------------------------------------
def shap_tree(model, X: pd.DataFrame, outdir, tag, max_display=20, nsample=3000):
    """SHAP for tree models: beeswarm + bar + dependence on the top feature."""
    import shap
    Xs = X.sample(min(nsample, len(X)), random_state=0)
    try:
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xs)
    except Exception as e:
        print(f"    [shap] skipped: {e}", flush=True)
        return None
    vals = sv[1] if isinstance(sv, list) and len(sv) == 2 else sv
    if isinstance(vals, list):
        vals = np.mean([np.abs(v) for v in vals], axis=0)
    if getattr(vals, "ndim", 2) == 3:                 # (n, feat, class)
        vals = np.abs(vals).mean(axis=2)

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(vals, Xs, max_display=max_display, show=False)
    plt.title(f"{tag} — SHAP beeswarm")
    _save(plt.gcf(), outdir, f"shap_beeswarm_{tag}")

    fig = plt.figure(figsize=(7, 5))
    shap.summary_plot(vals, Xs, plot_type="bar", max_display=max_display, show=False)
    plt.title(f"{tag} — SHAP mean |value|")
    _save(plt.gcf(), outdir, f"shap_bar_{tag}")

    imp = np.abs(vals).mean(0)
    order = np.argsort(-imp)
    return {"shap_importance": {X.columns[i]: float(imp[i]) for i in order}}


def lime_explain(predict_fn, X: pd.DataFrame, y, class_names, outdir, tag,
                 n_examples=4, seed=0):
    """LIME local explanations for a few correctly- and incorrectly-handled rows."""
    from lime.lime_tabular import LimeTabularExplainer
    rng = np.random.RandomState(seed)
    expl = LimeTabularExplainer(X.to_numpy(), feature_names=list(X.columns),
                                class_names=list(class_names), discretize_continuous=True,
                                random_state=seed)
    picks = rng.choice(len(X), min(n_examples, len(X)), replace=False)
    out = {}
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 3.2 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, picks):
        try:
            e = expl.explain_instance(X.iloc[i].to_numpy(), predict_fn,
                                      num_features=10,
                                      top_labels=1)
            lab = e.available_labels()[0]
            pairs = e.as_list(label=lab)
            names = [p[0] for p in pairs][::-1]
            weights = [p[1] for p in pairs][::-1]
            cols = [PALETTE[0] if w > 0 else PALETTE[3] for w in weights]
            ax.barh(range(len(names)), weights, color=cols)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(f"row {i} | true={class_names[int(y[i])]} | "
                         f"explained class={class_names[int(lab)]}", fontsize=9)
            ax.axvline(0, color="black", lw=0.8)
            out[str(i)] = {"true": class_names[int(y[i])],
                           "explained_class": class_names[int(lab)],
                           "weights": pairs}
        except Exception as ex:
            ax.text(0.5, 0.5, f"LIME failed: {ex}", ha="center")
    plt.tight_layout()
    _save(fig, outdir, f"lime_{tag}")
    return out


def permutation_importance_fig(model, X, y, outdir, tag, n_repeats=3, nsample=20000):
    from sklearn.inspection import permutation_importance
    idx = np.random.RandomState(0).choice(len(X), min(nsample, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx], y[idx], n_repeats=n_repeats,
                               random_state=0, scoring="f1_macro", n_jobs=-1)
    order = np.argsort(-r.importances_mean)[:20]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), r.importances_mean[order][::-1],
            xerr=r.importances_std[order][::-1], color=PALETTE[0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([X.columns[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("drop in macro-F1 when permuted")
    ax.set_title(f"{tag} — permutation importance")
    _save(fig, outdir, f"permutation_{tag}")
    return {X.columns[i]: float(r.importances_mean[i]) for i in order}


def plot_dt_attribution(field_names: Sequence[str], dt_mean: np.ndarray,
                        outdir, tag):
    """Mamba-specific: the S6 selection gate dt as intrinsic field attribution."""
    order = np.argsort(-dt_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), dt_mean[order][::-1], color=PALETTE[1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([field_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("mean selection gate  dt  (higher = field updates state more)")
    ax.set_title(f"{tag} — Mamba intrinsic dt attribution")
    _save(fig, outdir, f"dt_attribution_{tag}")
    return {field_names[i]: float(dt_mean[i]) for i in order}


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def save_metrics(all_metrics: Dict, outdir: str, name: str = "metrics"):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    rows = []
    for ds, models in all_metrics.items():
        if not isinstance(models, dict):
            continue
        for mdl, splits in models.items():
            if not isinstance(splits, dict):
                continue
            for sp, m in splits.items():
                if not isinstance(m, dict) or "accuracy" not in m:
                    continue
                row = {"dataset": ds, "model": mdl, "split": sp}
                row.update({k: v for k, v in m.items()
                            if isinstance(v, (int, float))})
                for cn, cv in m.get("per_class", {}).items():
                    for k, v in cv.items():
                        row[f"{cn}_{k}"] = v
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    return len(rows)
