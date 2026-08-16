"""Run federated MoE-Mamba across datasets sequentially, one at a time.

    python exp/fed_chain.py bots guide        # after AIT finishes
    python exp/fed_chain.py ait bots guide    # from scratch

Waits for any already-running fed_run.py to exit first, then runs each dataset
in turn, retrying a dataset once if it dies from the intermittent native crash
(0xC0000005 in torch_cpu.dll) that killed two earlier runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LOGS = os.path.join(BASE, "logs")


def pid_alive(pid: int) -> bool:
    """Windows-safe liveness check. `wmic` is absent on Windows 11, so use
    tasklist with a PID filter, which prints a 'No tasks' banner when dead."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            stderr=subprocess.DEVNULL, text=True)
        return str(pid) in out
    except Exception:
        return False


def main() -> int:
    argv = sys.argv[1:]
    wait_pid = None
    if argv and argv[0] == "--wait-pid":
        wait_pid = int(argv[1])
        argv = argv[2:]
    datasets = argv or ["ait", "bots", "guide"]
    os.makedirs(LOGS, exist_ok=True)

    waited = 0
    while wait_pid is not None and pid_alive(wait_pid):
        if waited % 300 == 0:
            print(f"[chain] waiting for pid {wait_pid} "
                  f"({waited // 60} min elapsed)", flush=True)
        time.sleep(30)
        waited += 30
    if wait_pid is not None:
        print(f"[chain] pid {wait_pid} finished, starting queue", flush=True)

    for ds in datasets:
        for attempt in (1, 2):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[chain] === {ds} attempt {attempt} {stamp} ===", flush=True)
            log = os.path.join(LOGS, f"fed_{ds}.log")
            mode = "a" if attempt > 1 else "w"
            with open(log, mode, encoding="utf-8") as fh:
                rc = subprocess.call(
                    [sys.executable, "-u", os.path.join("exp", "fed_run.py"), ds],
                    cwd=BASE, stdout=fh, stderr=subprocess.STDOUT)
            if rc == 0:
                print(f"[chain] {ds} complete", flush=True)
                break
            print(f"[chain] {ds} crashed rc={rc} "
                  f"(0x{rc & 0xFFFFFFFF:08X})", flush=True)
            time.sleep(10)
        else:
            print(f"[chain] {ds} failed twice — moving on", flush=True)

    print("\n[chain] all datasets done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
