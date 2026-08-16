"""BOTSv3 rich per-sourcetype feature extraction — LEAKAGE-SAFE by construction.

The labelling problem
---------------------
BOTSv3 ships no ground truth. We label an event malicious iff its `_raw` contains
a published IOC (C2 IP 45.77.53.176, hdoor.exe, the leaked AWS key, ...). Any
feature derived from `_raw` that *contains* those tokens therefore reconstructs
the label -> circular, exactly like GUIDE's `IncidentId` and AIT-ADS's `location`.

Masking with a sentinel does NOT fix this: "value == <MASKED>" is itself a
perfect predictor. The only sound fix is to never expose identifier-type fields.

So we extract BEHAVIOUR ONLY
---------------------------
    ports, byte/packet volumes, durations, protocol, action, event codes,
    DNS query/reply types, HTTP method/status, counts, structural stats
and we explicitly EXCLUDE
    IP addresses, domains, URLs, hostnames, file names/paths, hashes, users,
    process names, e-mail addresses, AWS keys, and timestamps
(the last because the attack occupies a narrow window, so time leaks the label).

The resulting research question is honest and non-trivial:
    "can behavioural telemetry alone flag the activity that threat intel caught?"
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CSV_DIR = r"D:\Drive D\Ajloun Papers\paper 57 tylor\extracted\botsv3_csv"

# ---- published BOTSv3 IOCs, used ONLY to build labels, never as features ----
IOCS = [
    "45.77.53.176", "104.207.83.63", "139.198.18.205", "35.153.154.221",
    "209.107.196.112", "82.102.18.111",
    "botsv3.ministerofmayhem.com",
    "hdoor.exe", "iexeplorer.exe", "definitelydontinvestigatethisfile.sh",
    "frothly-brewery-financial-planning-fy2019-draft.xlsm",
    "586ef56f4d8963dd546163ac31c865d7",
    "akiajogcdxj5nw5pxupa", "frothlywebcode",
    "hyunki1984@naver.com", "yunki1984@naver.com",
]
IOC_RE = re.compile("|".join(re.escape(x) for x in IOCS), re.IGNORECASE)

# behavioural feature schema shared by all sourcetypes (missing -> None)
CAT_FIELDS = ["sourcetype", "protocol", "action", "event_code", "dns_query_type",
              "dns_reply_code", "http_method", "http_status", "direction",
              "app", "os_event_type"]
NUM_FIELDS = ["src_port", "dest_port", "bytes", "bytes_in", "bytes_out",
              "packets", "packets_in", "packets_out", "duration", "count",
              "raw_len", "n_tokens", "n_digits", "src_port_is_ephemeral",
              "dest_port_is_wellknown"]

_KV = re.compile(r'(\w+)="([^"]*)"')
_ASA = re.compile(r"%ASA-\d+-(\d+):\s*(\w+)\s+(\w+)")
_WIN_EC = re.compile(r"EventCode=(\d+)")
_WIN_TYPE = re.compile(r"TaskCategory=([^\r\n]*)")
_SYS_EID = re.compile(r"<EventID>(\d+)</EventID>")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _port_feats(d):
    sp, dp = d.get("src_port"), d.get("dest_port")
    d["src_port_is_ephemeral"] = 1.0 if (sp is not None and sp >= 32768) else 0.0
    d["dest_port_is_wellknown"] = 1.0 if (dp is not None and dp < 1024) else 0.0
    return d


def parse_event(raw: str, sourcetype: str) -> Dict:
    """_raw -> behavioural features only. Never returns an identifier value."""
    d: Dict = {"sourcetype": sourcetype,
               "raw_len": float(len(raw)),
               "n_tokens": float(raw.count(" ") + raw.count(",")),
               "n_digits": float(sum(c.isdigit() for c in raw))}
    st = sourcetype.lower()

    # ---------- JSON-shaped stream:* and AWS ----------
    if raw.lstrip().startswith("{"):
        try:
            j = json.loads(raw)
        except json.JSONDecodeError:
            j = {}
        if j:
            d["src_port"] = _f(j.get("src_port"))
            d["dest_port"] = _f(j.get("dest_port"))
            d["bytes"] = _f(j.get("bytes") or j.get("sum(bytes)"))
            d["bytes_in"] = _f(j.get("bytes_in") or j.get("sum(bytes_in)"))
            d["bytes_out"] = _f(j.get("bytes_out") or j.get("sum(bytes_out)"))
            d["packets_in"] = _f(j.get("packets_in") or j.get("sum(packets_in)"))
            d["packets_out"] = _f(j.get("packets_out") or j.get("sum(packets_out)"))
            d["count"] = _f(j.get("count") or j.get("counter"))
            d["duration"] = _f(j.get("response_time") or j.get("sum(time_taken)"))
            d["app"] = j.get("app")
            d["protocol"] = j.get("protocol") or j.get("transport")
            # DNS: TYPE and REPLY CODE only (never the queried name)
            qt = j.get("query_type")
            d["dns_query_type"] = qt[0] if isinstance(qt, list) and qt else qt
            d["dns_reply_code"] = j.get("reply_code")
            mt = j.get("message_type")
            d["direction"] = (",".join(mt) if isinstance(mt, list) else mt)
            # CloudTrail: the API verb is behaviour, the ARN/IP is not
            if j.get("eventName"):
                d["action"] = j.get("eventName")
                d["os_event_type"] = j.get("eventType")
            # osquery: pack name is behavioural (which rule fired)
            if j.get("name") and "osquery" in st:
                d["action"] = j.get("name")
            return _port_feats(d)

    # ---------- VPC flow logs (space separated, fixed columns) ----------
    if "vpcflow" in st:
        p = raw.split()
        if len(p) >= 14:
            d["src_port"], d["dest_port"] = _f(p[5]), _f(p[6])
            d["protocol"] = p[7]
            d["packets"], d["bytes"] = _f(p[8]), _f(p[9])
            d["duration"] = (_f(p[11]) or 0) - (_f(p[10]) or 0)
            d["action"], d["os_event_type"] = p[12], p[13]
        return _port_feats(d)

    # ---------- Cisco ASA ----------
    if "cisco" in st:
        m = _ASA.search(raw)
        if m:
            d["event_code"], d["action"], d["protocol"] = m.group(1), m.group(2), m.group(3)
        mb = re.search(r"bytes (\d+)", raw)
        if mb:
            d["bytes"] = _f(mb.group(1))
        md = re.search(r"duration (\d+):(\d+):(\d+)", raw)
        if md:
            d["duration"] = 3600 * _f(md.group(1)) + 60 * _f(md.group(2)) + _f(md.group(3))
        for m2 in re.finditer(r"/(\d{1,5})\b", raw):
            v = _f(m2.group(1))
            if d.get("src_port") is None:
                d["src_port"] = v
            elif d.get("dest_port") is None:
                d["dest_port"] = v
        return _port_feats(d)

    # ---------- Windows Security / Sysmon ----------
    if "wineventlog" in st or "sysmon" in st:
        m = _WIN_EC.search(raw) or _SYS_EID.search(raw)
        if m:
            d["event_code"] = m.group(1)
        mt = _WIN_TYPE.search(raw)
        if mt:
            d["os_event_type"] = mt.group(1).strip()
        return _port_feats(d)

    # ---------- key="value" syslog (nvzFlow etc.) ----------
    kv = dict(_KV.findall(raw))
    if kv:
        d["src_port"], d["dest_port"] = _f(kv.get("sp")), _f(kv.get("dp"))
        d["protocol"] = kv.get("pr")
        d["bytes_in"], d["bytes_out"] = _f(kv.get("bi")), _f(kv.get("bo"))
        d["os_event_type"] = kv.get("Type")
    return _port_feats(d)


def load_bots_rich(per_file_cap: int | None = None, verbose: bool = True
                   ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (records, y, host, sourcetype)."""
    recs: List[Dict] = []
    ys: List[int] = []
    hosts: List[str] = []
    sts: List[str] = []

    def clean(v):
        v = v or ""
        return v.split("::", 1)[1] if "::" in v else v

    for fp in sorted(glob.glob(os.path.join(CSV_DIR, "*.csv"))):
        if os.path.getsize(fp) < 5000:
            continue
        kept = 0
        with open(fp, encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            try:
                hdr = next(r)
            except StopIteration:
                continue
            c = {n: i for i, n in enumerate(hdr)}
            need = [c.get("_raw"), c.get("host"), c.get("sourcetype")]
            if any(x is None for x in need):
                continue
            ir, ih, iS = need
            for rec in r:
                if len(rec) <= max(need):
                    continue
                raw = rec[ir]
                st = clean(rec[iS])
                recs.append(parse_event(raw, st))
                ys.append(1 if IOC_RE.search(raw) else 0)
                hosts.append(clean(rec[ih]).lower())
                sts.append(st)
                kept += 1
                if per_file_cap and kept >= per_file_cap:
                    break
        if verbose:
            print(f"  {os.path.basename(fp):42s} total={len(recs):,}", flush=True)

    y = np.asarray(ys, dtype=np.int64)
    if verbose:
        print(f"  BOTSv3-rich rows={len(recs):,} malicious={int(y.sum()):,} "
              f"rate={y.mean():.5f}", flush=True)
    return recs, y, np.asarray(hosts), np.asarray(sts)
