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


def write_test_parameters(
    original_text,
    map_name,
    mode,
    loss,
    seed,
    retransmission_seed,
    retransmission,
    episodes,
    map_offset,
    simulation_seed,
    rlmr_v2_map_shield,
    rlmr_v2_profile,
    meta_agents,
    v1_q_table,
    v2_q_table,
    v2_1_q_table,
):
    text = original_text
    text = replace_assignment(text, "TEST_SET_NAME", repr(map_name))
    text = replace_assignment(text, "TEST_MAP_OFFSET", repr(int(map_offset)))
    text = replace_assignment(text, "TEST_RANDOM_SEED", repr(int(simulation_seed)))
    text = replace_assignment(text, "NUM_TEST", repr(int(episodes)))
    if meta_agents is not None:
        text = replace_assignment(text, "NUM_META_AGENT", repr(int(meta_agents)))
    text = replace_assignment(text, "ENABLE_PACKET_LOSS", "False")
    text = replace_assignment(text, "PACKET_LOSS_PROB", repr(0.0))
    text = replace_assignment(text, "PACKET_LOSS_SEED", repr(0))
    text = replace_assignment(text, "ENABLE_MESSAGE_LOSS", "True")
    text = replace_assignment(text, "MESSAGE_LOSS_MODE", repr(mode))
    text = replace_assignment(text, "MESSAGE_LOSS_PROB", repr(float(loss)))
    text = replace_assignment(text, "MESSAGE_LOSS_SEED", repr(int(seed)))
    text = replace_assignment(text, "RETRANSMISSION_LOSS_SEED", repr(int(retransmission_seed)))
    text = replace_assignment(text, "ENABLE_PRIORITY_RETRANSMISSION", str(retransmission != "off"))
    text = replace_assignment(text, "RLMR_TRAIN", "False")
    text = replace_assignment(text, "RLMR_V2_TRAIN", "False")
    text = replace_assignment(text, "RLMR_V2_MAP_SHIELD", str(rlmr_v2_map_shield))
    text = replace_assignment(text, "RLMR_V2_PROFILE", repr(rlmr_v2_profile))
    text = replace_assignment(text, "RLMR_Q_TABLE_PATH", repr(v1_q_table))
    text = replace_assignment(text, "RLMR_V2_Q_TABLE_PATH", repr(v2_q_table))
    text = replace_assignment(text, "RLMR_V2_1_Q_TABLE_PATH", repr(v2_1_q_table))
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


def run_one(
    map_name,
    mode,
    loss,
    seed,
    retransmission_seed,
    retransmission,
    map_offset,
    simulation_seed,
    output_tag,
    rlmr_v2_map_shield,
    rlmr_v2_profile,
    overwrite,
):
    output_policy = "rlmr_v2_1" if retransmission == "rlmr_v2" and rlmr_v2_profile == "v2_1" else retransmission
    retrans_suffix = "" if retransmission == "off" else f"_{output_policy}_retrans"
    if retransmission == "rlmr_v2" and not rlmr_v2_map_shield:
        retrans_suffix += "_no_shield"
    tag_suffix = f"_{output_tag}" if output_tag else ""
    target = LOG_DIR / f"{map_name}_msg_{mode}_{loss_suffix(loss)}{retrans_suffix}{tag_suffix}.csv"
    skipped_target = LOG_DIR / f"{map_name}_msg_{mode}_{loss_suffix(loss)}{retrans_suffix}{tag_suffix}_skipped.csv"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Use --overwrite to replace it.")
    if skipped_target.exists() and not overwrite:
        raise FileExistsError(f"{skipped_target} already exists. Use --overwrite to replace it.")

    old_files = set(LOG_DIR.glob("data_*.csv"))
    old_skipped_files = set(LOG_DIR.glob("skipped_*.csv"))
    start_time = time.time()

    print("=" * 80, flush=True)
    print(
        f"Running map={map_name}, map_offset={map_offset}, simulation_seed={simulation_seed}, "
        f"message_mode={mode}, "
        f"message_loss={loss:.1f}, message_seed={seed}, retransmission={retransmission}, "
        f"retransmission_seed={retransmission_seed}, profile={rlmr_v2_profile}, "
        f"map_shield={rlmr_v2_map_shield}, "
        f"output={target.name}",
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
        "--retransmission-seed",
        type=int,
        default=100000,
        help="Base retransmission-loss RNG seed. Default: 100000.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes attempted per setting. Default: 100.",
    )
    parser.add_argument(
        "--map-offset",
        type=int,
        default=0,
        help="Start map index for each setting. Default: 0.",
    )
    parser.add_argument(
        "--simulation-seed",
        type=int,
        default=0,
        help="Base seed for episode-level Python, NumPy, and PyTorch randomness.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for output files, for example heldout_seed1000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing message-loss CSV files.",
    )
    parser.add_argument(
        "--retransmission",
        default="off",
        choices=["off", "priority", "equal", "adaptive", "rlmr", "rlmr_v2"],
        help="Enable pending-message retransmission. Default: off.",
    )
    parser.add_argument(
        "--no-map-shield",
        action="store_true",
        help="Disable the RLMR-v2 critical-map safety constraint.",
    )
    parser.add_argument("--rlmr-v2-profile", choices=["v2", "v2_1"], default="v2")
    parser.add_argument("--meta-agents", type=int, default=None,
                        help="Override Ray worker count; omit to keep test_parameter.py unchanged.")
    parser.add_argument("--v1-q-table", default="mar_inference/rlmr_q_table.json")
    parser.add_argument("--v2-q-table", default="mar_inference/rlmr_v2_q_table.json")
    parser.add_argument("--v2-1-q-table", default="mar_inference/rlmr_v2_1_q_table.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.map_offset < 0:
        raise ValueError("--map-offset must be non-negative")
    if args.meta_agents is not None and args.meta_agents <= 0:
        raise ValueError("--meta-agents must be positive")
    if args.output_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        raise ValueError("--output-tag may contain only letters, numbers, underscores, and hyphens")
    original_text = PARAM_PATH.read_text(encoding="utf-8")

    try:
        run_index = 0
        for map_name in args.maps:
            for mode in args.modes:
                for loss in args.losses:
                    seed = args.seed + run_index
                    retransmission_seed = args.retransmission_seed + run_index
                    write_test_parameters(
                        original_text,
                        map_name,
                        mode,
                        loss,
                        seed,
                        retransmission_seed,
                        args.retransmission,
                        args.episodes,
                        args.map_offset,
                        args.simulation_seed,
                        not args.no_map_shield,
                        args.rlmr_v2_profile,
                        args.meta_agents,
                        args.v1_q_table,
                        args.v2_q_table,
                        args.v2_1_q_table,
                    )
                    run_one(
                        map_name,
                        mode,
                        loss,
                        seed,
                        retransmission_seed,
                        args.retransmission,
                        args.map_offset,
                        args.simulation_seed,
                        args.output_tag,
                        not args.no_map_shield,
                        args.rlmr_v2_profile,
                        args.overwrite,
                    )
                    run_index += 1
    finally:
        PARAM_PATH.write_text(original_text, encoding="utf-8")
        print(f"Restored {PARAM_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
