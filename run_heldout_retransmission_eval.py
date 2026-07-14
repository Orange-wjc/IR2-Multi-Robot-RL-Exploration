#!/usr/bin/env python3
"""Run the held-out retransmission comparison with a frozen RLMR Q table."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SWEEP_SCRIPT = ROOT / "run_message_loss_sweep.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run held-out NoRetrans, PriorityRetrans, and frozen RLMR evaluations."
    )
    parser.add_argument("--map", default="corridor", choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--policies", nargs="+", default=["off", "priority", "rlmr"],
                        choices=["off", "priority", "equal", "adaptive", "rlmr"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--map-offset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--retransmission-seed", type=int, default=2000)
    parser.add_argument("--output-tag", default="heldout_seed1000")
    parser.add_argument("--q-table", default="mar_inference/rlmr_q_table.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    q_table = ROOT / args.q_table
    if not q_table.is_file():
        raise FileNotFoundError(f"RLMR Q table not found: {q_table}")

    frozen_hash = file_sha256(q_table)
    print(f"Frozen Q-table SHA-256: {frozen_hash}", flush=True)

    for policy in args.policies:
        command = [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--maps", args.map,
            "--modes", "random",
            "--losses", *[str(loss) for loss in args.losses],
            "--retransmission", policy,
            "--episodes", str(args.episodes),
            "--map-offset", str(args.map_offset),
            "--seed", str(args.seed),
            "--retransmission-seed", str(args.retransmission_seed),
            "--output-tag", args.output_tag,
        ]
        if args.overwrite:
            command.append("--overwrite")

        print("=" * 80, flush=True)
        print(f"Held-out evaluation policy={policy}", flush=True)
        print(" ".join(command), flush=True)
        print("=" * 80, flush=True)

        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            current_hash = file_sha256(q_table)
            if current_hash != frozen_hash:
                raise RuntimeError(
                    "RLMR Q table changed during evaluation: "
                    f"expected {frozen_hash}, got {current_hash}"
                )
            print(f"Q-table unchanged after policy={policy}: {current_hash}", flush=True)

    print("Held-out retransmission evaluation completed.", flush=True)


if __name__ == "__main__":
    main()
