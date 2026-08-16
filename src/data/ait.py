"""AIT-ADS loader -> unified (field,value) records.

Three detector schemas are mapped onto ONE field vocabulary, which is exactly
the heterogeneity the tokenizer is designed to absorb:

  Wazuh    : rule.id, rule.level, rule.groups, decoder.name, agent.name, location
  Suricata : (arrives wrapped inside Wazuh) -> data.alert.signature/category/severity
  AMiner   : AnalysisComponentName, AnalysisComponentType, #AffectedLogAtomPaths

Labels: phase-window (alert inside an attack window -> 1). Known to over-label
benign noise during attack periods; alert-level labels from the ait-aecid repo
should replace these before publication.
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCENARIOS = ["fox", "harrison", "russellmitchell", "santos",
             "shaw", "wardbeck", "wheeler", "wilson"]

AIT_DIR = r"D:\Drive D\Ajloun Papers\paper 57 tylor\extracted\ait_ads"
LABELS_ZIP = r"D:\Drive D\Ajloun Papers\paper 57 tylor\8263181.zip"

CAT_FIELDS = ["detector", "signature", "rule_group", "decoder", "agent",
              "location", "component_type", "suri_category"]
NUM_FIELDS = ["severity", "n_paths", "log_lines", "firedtimes"]


def load_windows() -> Dict[str, List[Tuple[str, float, float]]]:
    win = {s: [] for s in SCENARIOS}
    with zipfile.ZipFile(LABELS_ZIP) as z, z.open("labels.csv") as f:
        for row in csv.DictReader(io.TextIOWrapper(f, "utf-8")):
            win[row["scenario"]].append(
                (row["attack"], float(row["start"]), float(row["end"])))
    return win


def _label(win, scn, epoch) -> int:
    for _, s, e in win[scn]:
        if s <= epoch <= e:
            return 1
    return 0


def load_ait(cap_wazuh_per_scenario: int = 30_000, verbose: bool = True):
    """Returns (records, y, scenario) — records are plain dicts."""
    win = load_windows()
    recs: List[Dict] = []
    ys: List[int] = []
    scns: List[str] = []

    for scn in SCENARIOS:
        # ---- AMiner (parse all) ----
        p = os.path.join(AIT_DIR, f"{scn}_aminer.json")
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ld = a.get("LogData", {})
                ts = (ld.get("Timestamps") or [None])[0]
                if ts is None:
                    continue
                comp = a.get("AnalysisComponent", {})
                recs.append({
                    "detector": "aminer",
                    "signature": comp.get("AnalysisComponentName"),
                    "component_type": comp.get("AnalysisComponentType"),
                    "n_paths": len(comp.get("AffectedLogAtomPaths") or []),
                    "log_lines": ld.get("LogLinesCount"),
                    "location": (ld.get("LogResources") or [None])[0],
                })
                ys.append(_label(win, scn, float(ts))); scns.append(scn)

        # ---- Wazuh / Suricata (strided sample) ----
        p = os.path.join(AIT_DIR, f"{scn}_wazuh.json")
        est = max(1, os.path.getsize(p) // 900)
        step = max(1, est // cap_wazuh_per_scenario)
        kept = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i % step:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    ts = pd.Timestamp(a.get("timestamp") or a.get("@timestamp")).timestamp()
                except Exception:
                    continue
                rule = a.get("rule", {}) or {}
                groups = rule.get("groups") or []
                data = a.get("data", {}) or {}
                sa = (data.get("alert") or {}) if isinstance(data, dict) else {}
                is_suri = bool(sa) or any("suricata" in str(g) or "ids" in str(g) for g in groups)
                recs.append({
                    "detector": "suricata" if is_suri else "wazuh",
                    "signature": sa.get("signature") or rule.get("id"),
                    "rule_group": groups[0] if groups else None,
                    "decoder": (a.get("decoder", {}) or {}).get("name"),
                    "agent": (a.get("agent", {}) or {}).get("name"),
                    "location": a.get("location"),
                    "suri_category": sa.get("category"),
                    "severity": sa.get("severity", rule.get("level")),
                    "firedtimes": rule.get("firedtimes"),
                })
                ys.append(_label(win, scn, ts)); scns.append(scn)
                kept += 1
                if kept >= cap_wazuh_per_scenario:
                    break
        if verbose:
            print(f"  {scn:16s} total={len(recs):,}", flush=True)

    return recs, np.asarray(ys, dtype=np.int64), np.asarray(scns)
