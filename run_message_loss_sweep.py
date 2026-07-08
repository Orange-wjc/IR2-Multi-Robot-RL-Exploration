#!/usr/bin/env python3
"""Run message-level dropout inference sweeps with the released IR2 checkpoint.

Default run:
    python run_message_loss_sweep.py

Smoke test a single setting after lowering NUM_TEST manually:
    python run_message_loss_sweep.py --maps corridor --modes pose --losses 0.3
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARAM_PATH = ROOT / "test_parameter.py"
LOG_DIR = ROOT / "mar_inference" / "test_results" / "log"


def loss_suffix(loss):
    return f"{int(round(loss * 10)):02d}"


def replace_assignment(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name}={value}"
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return new_text


def write_test_parameters(original_text, map_name, mode, loss, seed, retransmission):
    text = original_text
    text = replace_assignment(text, "TEST_SET_NAME", repr(map_name))
    text = replace_assignment(text, "ENABLE_PACKET_LOSS", "False")
    text = replace_assignment(text, "PACKET_LOSS_PROB", repr(0.0))
    text = replace_assignment(text, "PACKET_LOSS_SEED", repr(0))
    text = replace_assignment(text, "ENABLE_MESSAGE_LOSS", "True")
    text = replace_assignment(text, "MESSAGE_LOSS_MODE", repr(mode))
    text = replace_assignment(text, "MESSAGE_LOSS_PROB", repr(float(loss)))
    text = replace_assignment(text, "MESSAGE_LOSS_SEED", repr(int(seed)))
    text = replace_assignment(text, "ENABLE_PRIORITY_RETRANSMISSION", str(retransmission != "off"))
    if retransmission != "off":
        text = replace_assignment(text, "RETRANSMISSION_POLICY", repr(retransmission))
    PARAM_PATH.write_text(text, encoding="utf-8")


def latest_new_csv(start_time, old_files, pattern):
    candidates = []
    for path in LOG_DIR.glob(pattern):
        if path in old_files:
            continue
        if path.stat().st_mtime >= start_time - 1:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_one(map_name, mode, loss, seed, retransmission, overwrite):
    retrans_suffix = "" if retransmission == "off" else f"_{retransmission}_retrans"
    target = LOG_DIR / f"{map_name}_msg_{mode}_{loss_suffix(loss)}{retrans_suffix}.csv"
    skipped_target = LOG_DIR / f"{map_name}_msg_{mode}_{loss_suffix(loss)}{retrans_suffix}_skipped.csv"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Use --overwrite to replace it.")
    if skipped_target.exists() and not overwrite:
        raise FileExistsError(f"{skipped_target} already exists. Use --overwrite to replace it.")

    old_files = set(LOG_DIR.glob("data_*.csv"))
    old_skipped_files = set(LOG_DIR.glob("skipped_*.csv"))
    start_time = time.time()

    print("=" * 80, flush=True)
    print(
        f"Running map={map_name}, message_mode={mode}, message_loss={loss:.1f}, retransmission={retransmission}, output={target.name}",
        flush=True,
    )
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env.setdefault("RAY_DEDUP_LOGS", "0")
    subprocess.run([sys.executable, "test_driver.py"], cwd=ROOT, env=env, check=True)

    new_csv = latest_new_csv(start_time, old_files, "data_*.csv")
    if new_csv is None:
        raise RuntimeError("test_driver.py finished but no new data_*.csv was found")
    new_skipped_csv = latest_new_csv(start_time, old_skipped_files, "skipped_*.csv")
    if new_skipped_csv is None:
        raise RuntimeError("test_driver.py finished but no new skipped_*.csv was found")

    if target.exists() and overwrite:
        target.unlink()
    if skipped_target.exists() and overwrite:
        skipped_target.unlink()
    new_csv.rename(target)
    new_skipped_csv.rename(skipped_target)
    print(f"Saved {target}", flush=True)
    print(f"Saved {skipped_target}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run IR2 message-level dropout sweeps.")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["corridor", "hybrid"],
        choices=["corridor", "hybrid", "complex"],
        help="Map sets to run. Default: corridor hybrid.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["map", "pose", "graph", "all"],
        choices=["map", "pose", "graph", "all", "random"],
        help="Message types to drop. Default: map pose graph all.",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="Message loss probabilities. Default: 0.1 0.2 0.3 0.5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base message-loss RNG seed. Default: 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing message-loss CSV files.",
    )
    parser.add_argument(
        "--retransmission",
        default="off",
        choices=["off", "priority", "equal", "adaptive"],
        help="Enable pending-message retransmission. Default: off.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    original_text = PARAM_PATH.read_text(encoding="utf-8")

    try:
        run_index = 0
        for map_name in args.maps:
            for mode in args.modes:
                for loss in args.losses:
                    seed = args.seed + run_index
                    write_test_parameters(original_text, map_name, mode, loss, seed, args.retransmission)
                    run_one(map_name, mode, loss, seed, args.retransmission, args.overwrite)
                    run_index += 1
    finally:
        PARAM_PATH.write_text(original_text, encoding="utf-8")
        print(f"Restored {PARAM_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
