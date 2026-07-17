#!/usr/bin/env python3
"""Train the RLMR-v2 TD Q-learning communication controller."""

import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARAM_PATH = ROOT / "test_parameter.py"
DEFAULT_Q_TABLES = {
    "v2": "mar_inference/rlmr_v2_q_table.json",
    "v2_1": "mar_inference/rlmr_v2_1_q_table.json",
}


def replace_assignment(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    replacement = f"{name}={value}"
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return new_text


def write_training_parameters(original_text, setting, args, q_table_path):
    text = original_text
    text = replace_assignment(text, "TEST_SET_NAME", repr(setting["map"]))
    text = replace_assignment(text, "TEST_MAP_OFFSET", repr(args.map_offset))
    text = replace_assignment(text, "NUM_TEST", repr(args.episodes_per_setting))
    text = replace_assignment(text, "NUM_META_AGENT", repr(1))
    text = replace_assignment(text, "ENABLE_PACKET_LOSS", "False")
    text = replace_assignment(text, "PACKET_LOSS_PROB", repr(0.0))
    text = replace_assignment(text, "PACKET_LOSS_SEED", repr(0))
    text = replace_assignment(text, "ENABLE_MESSAGE_LOSS", "True")
    text = replace_assignment(text, "MESSAGE_LOSS_MODE", repr("random"))
    text = replace_assignment(text, "MESSAGE_LOSS_PROB", repr(setting["loss"]))
    text = replace_assignment(text, "MESSAGE_LOSS_SEED", repr(setting["message_seed"]))
    text = replace_assignment(text, "RETRANSMISSION_LOSS_SEED", repr(setting["retransmission_seed"]))
    text = replace_assignment(text, "ENABLE_PRIORITY_RETRANSMISSION", "True")
    text = replace_assignment(text, "RETRANSMISSION_POLICY", repr("rlmr_v2"))
    text = replace_assignment(text, "RLMR_TRAIN", "False")
    text = replace_assignment(text, "RLMR_V2_TRAIN", "True")
    text = replace_assignment(text, "RLMR_V2_PROFILE", repr(args.profile))
    q_table_parameter = "RLMR_V2_Q_TABLE_PATH" if args.profile == "v2" else "RLMR_V2_1_Q_TABLE_PATH"
    text = replace_assignment(text, q_table_parameter, repr(str(q_table_path)))
    text = replace_assignment(text, "RLMR_V2_MAP_SHIELD", str(not args.no_map_shield))
    PARAM_PATH.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RLMR-v2 with balanced, shuffled message-loss settings."
    )
    parser.add_argument("--maps", nargs="+", default=["corridor"],
                        choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--episodes-per-setting", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--map-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retransmission-seed", type=int, default=100000)
    parser.add_argument("--profile", choices=["v2", "v2_1"], default="v2",
                        help="Controller profile to train. Default: v2 for backward compatibility.")
    parser.add_argument("--q-table", default=None,
                        help="Output Q table. Defaults to a profile-specific path.")
    parser.add_argument("--no-map-shield", action="store_true",
                        help="Train the pure TD policy without the critical-map safety constraint.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_epoch_settings(args, epoch):
    settings = []
    for map_name in args.maps:
        for loss_index, loss in enumerate(args.losses):
            settings.append({
                "map": map_name,
                "loss": float(loss),
                "message_seed": args.seed + epoch * len(args.losses) + loss_index,
                "retransmission_seed": args.retransmission_seed + epoch * len(args.losses) + loss_index,
            })
    random.Random(args.seed + epoch).shuffle(settings)
    return settings


def main():
    args = parse_args()
    if args.episodes_per_setting <= 0 or args.epochs <= 0:
        raise ValueError("--episodes-per-setting and --epochs must be positive")
    if args.map_offset < 0:
        raise ValueError("--map-offset must be non-negative")
    if any(loss <= 0 or loss > 1 for loss in args.losses):
        raise ValueError("--losses values must be in (0, 1]")

    q_table_name = args.q_table or DEFAULT_Q_TABLES[args.profile]
    if args.no_map_shield and args.q_table is None:
        q_table_name = f"mar_inference/rlmr_{args.profile}_no_shield_q_table.json"
    q_table_path = (ROOT / q_table_name).resolve()
    original_text = PARAM_PATH.read_text(encoding="utf-8")
    total_settings = args.epochs * len(args.maps) * len(args.losses)
    completed_settings = 0

    try:
        for epoch in range(args.epochs):
            for setting in make_epoch_settings(args, epoch):
                completed_settings += 1
                print("=" * 80, flush=True)
                print(
                    f"RLMR-{args.profile} setting {completed_settings}/{total_settings}: "
                    f"epoch={epoch + 1}/{args.epochs}, map={setting['map']}, "
                    f"loss={setting['loss']:.1f}, episodes={args.episodes_per_setting}, "
                    f"map_shield={not args.no_map_shield}",
                    flush=True,
                )
                print(f"Q table: {q_table_path}", flush=True)
                print("=" * 80, flush=True)
                if args.dry_run:
                    continue

                write_training_parameters(original_text, setting, args, q_table_path)
                env = os.environ.copy()
                env.setdefault("RAY_DEDUP_LOGS", "0")
                subprocess.run([sys.executable, "test_driver.py"], cwd=ROOT, env=env, check=True)
    finally:
        if not args.dry_run:
            PARAM_PATH.write_text(original_text, encoding="utf-8")
            print(f"Restored {PARAM_PATH.name}", flush=True)

    print(f"RLMR-{args.profile} training complete. Q table: {q_table_path}", flush=True)


if __name__ == "__main__":
    main()
