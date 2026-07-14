#!/usr/bin/env python3
"""Train the lightweight RLMR communication controller.

This script keeps the released IR2 exploration checkpoint fixed. It only enables
RLMR_TRAIN so the tabular communication controller updates its Q table while
test_driver.py runs inference episodes under random message loss.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARAM_PATH = ROOT / "test_parameter.py"


def replace_assignment(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name}={value}"
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return new_text


def write_training_parameters(original_text, map_name, loss, seed, episodes, q_table_path):
    text = original_text
    text = replace_assignment(text, "TEST_SET_NAME", repr(map_name))
    text = replace_assignment(text, "TEST_MAP_OFFSET", repr(0))
    text = replace_assignment(text, "NUM_TEST", repr(int(episodes)))
    text = replace_assignment(text, "NUM_META_AGENT", repr(1))
    text = replace_assignment(text, "ENABLE_PACKET_LOSS", "False")
    text = replace_assignment(text, "PACKET_LOSS_PROB", repr(0.0))
    text = replace_assignment(text, "PACKET_LOSS_SEED", repr(0))
    text = replace_assignment(text, "ENABLE_MESSAGE_LOSS", "True")
    text = replace_assignment(text, "MESSAGE_LOSS_MODE", repr("random"))
    text = replace_assignment(text, "MESSAGE_LOSS_PROB", repr(float(loss)))
    text = replace_assignment(text, "MESSAGE_LOSS_SEED", repr(int(seed)))
    text = replace_assignment(text, "RETRANSMISSION_LOSS_SEED", repr(int(seed + 100000)))
    text = replace_assignment(text, "ENABLE_PRIORITY_RETRANSMISSION", "True")
    text = replace_assignment(text, "RETRANSMISSION_POLICY", repr("rlmr"))
    text = replace_assignment(text, "RLMR_TRAIN", "True")
    text = replace_assignment(text, "RLMR_Q_TABLE_PATH", repr(str(q_table_path)))
    PARAM_PATH.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RLMR communication controller.")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["corridor"],
        choices=["corridor", "hybrid", "complex"],
        help="Map sets used for RLMR training. Default: corridor.",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="Random message loss probabilities. Default: 0.1 0.2 0.3 0.5.",
    )
    parser.add_argument(
        "--episodes-per-setting",
        type=int,
        default=50,
        help="Training episodes per map/loss setting. Default: 50.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many times to sweep all map/loss settings. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Default: 0.",
    )
    parser.add_argument(
        "--q-table",
        default="mar_inference/rlmr_q_table.json",
        help="Output Q-table path. Default: mar_inference/rlmr_q_table.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    original_text = PARAM_PATH.read_text(encoding="utf-8")
    q_table_path = ROOT / args.q_table

    try:
        run_index = 0
        for epoch in range(args.epochs):
            for map_name in args.maps:
                for loss in args.losses:
                    seed = args.seed + run_index
                    write_training_parameters(
                        original_text,
                        map_name,
                        loss,
                        seed,
                        args.episodes_per_setting,
                        q_table_path,
                    )
                    print("=" * 80, flush=True)
                    print(
                        f"Training RLMR epoch={epoch + 1}/{args.epochs}, map={map_name}, loss={loss:.1f}, episodes={args.episodes_per_setting}",
                        flush=True,
                    )
                    print("=" * 80, flush=True)
                    env = os.environ.copy()
                    env.setdefault("RAY_DEDUP_LOGS", "0")
                    subprocess.run([sys.executable, "test_driver.py"], cwd=ROOT, env=env, check=True)
                    run_index += 1
    finally:
        PARAM_PATH.write_text(original_text, encoding="utf-8")
        print(f"Restored {PARAM_PATH.name}", flush=True)
        print(f"RLMR Q table: {q_table_path}", flush=True)


if __name__ == "__main__":
    main()
