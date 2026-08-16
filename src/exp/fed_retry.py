"""Run fed_run.py for one dataset, retrying across D: volume dropouts.

The D: volume has vanished mid-run three times in this project. Windows logs
Application event 1005 ("Windows cannot access the file ... the disk is
missing") and kills python with no traceback, so the run just stops. The volume
has recovered on its own every time.

This wrapper waits for D: to come back and relaunches; fed_run.py's own resume
logic skips algorithms already written to results/fed_<name>.json, so a dropout
costs at most the algorithm that was in flight.

    python exp/fed_retry.py guide 6
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
PROBE = os.path.join(BASE, "logs", "_disk_probe.tmp")


def disk_ok() -> bool:
    try:
        with open(PROBE, "w", encoding="ascii") as fh:
            fh.write("ok")
        os.remove(PROBE)
        return True
    except OSError:
        return False


def wait_for_disk(max_wait: int = 1800) -> bool:
    waited = 0
    while waited < max_wait:
        if disk_ok():
            if waited:
                print(f"[retry] D: back after {waited}s", flush=True)
            return True
        if waited % 60 == 0:
            print(f"[retry] D: unavailable, waiting ({waited}s)", flush=True)
        time.sleep(15)
        waited += 15
    return False


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "guide"
    attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    log = os.path.join(BASE, "logs", f"fed_{ds}5.log")

    for k in range(1, attempts + 1):
        if not wait_for_disk():
            print("[retry] D: did not return within 30 min, giving up", flush=True)
            return 2
        print(f"[retry] === {ds} attempt {k}/{attempts} "
              f"{time.strftime('%H:%M:%S')} ===", flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== attempt {k} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            fh.flush()
            rc = subprocess.call(
                [sys.executable, "-u", os.path.join("exp", "fed_run.py"), ds],
                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0:
            print(f"[retry] {ds} completed cleanly", flush=True)
            return 0
        print(f"[retry] attempt {k} died rc={rc} "
              f"(0x{rc & 0xFFFFFFFF:08X}) — resuming from disk", flush=True)
        time.sleep(20)

    print(f"[retry] {ds} still incomplete after {attempts} attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
