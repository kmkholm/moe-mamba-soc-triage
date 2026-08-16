"""GUIDE loader — FULL 38-column feature set (the published protocol).

This is the feature set that produced LightGBM 0.9887 / Mamba 0.9803 in B1/B2:
every usable column is kept, INCLUDING the identifier columns
`Id`, `OrgId`, `IncidentId`, `AlertId` and the entity identifiers.

It matches what the published work does — the GUIDE paper feeds
`OrganizationId` as a feature, and the Kaggle replication that reports
0.91/0.90 feeds both `OrgId` and `IncidentId`.

Companion to `data/guide.py`, which keeps only the semantic fields
(no identifiers) and scores ~0.79. Use whichever protocol you are reporting.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TRAIN_CSV = r"D:\Drive D\Ajloun Papers\paper 57 tylor\extracted\guide\GUIDE_Train.csv"

CLASSES = ["BenignPositive", "FalsePositive", "TruePositive"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# columns never used as features
NON_FEATURES = {"IncidentGrade", "Timestamp", "MitreTechniques", "Usage"}
# derived numerics
NUM_FIELDS = ["hour", "day", "dayofweek", "month", "mitre_n"]


def load_guide_full(nrows: int | None = 1_500_000, verbose: bool = True
                    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[str]]:
    """Returns (records, y, incident_groups, cat_field_names)."""
    df = pd.read_csv(TRAIN_CSV, nrows=nrows, low_memory=False)
    df = df.dropna(subset=["IncidentGrade"])
    df = df[df["IncidentGrade"].isin(CLASSES)].reset_index(drop=True)

    y = df["IncidentGrade"].map(CLASS_TO_ID).to_numpy().astype(np.int64)
    groups = (df["OrgId"].astype(str) + "|" + df["IncidentId"].astype(str)).to_numpy()

    # drop columns that are >50% missing (published preprocessing rule)
    miss = df.isna().mean()
    drop = set(NON_FEATURES) | {c for c in df.columns if miss[c] > 0.5}
    dropped = sorted(c for c in df.columns if c in drop and c not in NON_FEATURES)

    ts = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    num = {
        "hour": ts.dt.hour.fillna(-1).to_numpy(),
        "day": ts.dt.day.fillna(-1).to_numpy(),
        "dayofweek": ts.dt.dayofweek.fillna(-1).to_numpy(),
        "month": ts.dt.month.fillna(-1).to_numpy(),
        "mitre_n": df["MitreTechniques"].fillna("").astype(str)
                     .str.count(r"T\d{4}").to_numpy(),
    }

    cat_fields = [c for c in df.columns if c not in drop]
    cols = {c: df[c].to_numpy() for c in cat_fields}

    recs: List[Dict] = []
    n = len(df)
    for i in range(n):
        r = {c: (None if pd.isna(v[i]) else v[i]) for c, v in cols.items()}
        for k, v in num.items():
            r[k] = float(v[i])
        recs.append(r)

    if verbose:
        print(f"  GUIDE[full] rows={n:,} cat_fields={len(cat_fields)} "
              f"num_fields={len(NUM_FIELDS)} total={len(cat_fields)+len(NUM_FIELDS)}",
              flush=True)
        print(f"  dropped >50% missing: {dropped}", flush=True)
        print(f"  IDENTIFIERS INCLUDED: "
              f"{[c for c in ['Id','OrgId','IncidentId','AlertId'] if c in cat_fields]}",
              flush=True)
        print(f"  class dist "
              f"{pd.Series(y).value_counts(normalize=True).round(4).to_dict()}",
              flush=True)
    return recs, y, groups, cat_fields
