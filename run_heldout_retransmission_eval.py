#!/usr/bin/env python3
"""Run held-out retransmission comparisons with frozen RLMR Q tables."""

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
        description="Run held-out retransmission evaluations with frozen RLMR-v1/v2 tables."
    )
    parser.add_argument("--map", default="corridor", choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--policies", nargs="+", default=["off", "priority", "rlmr"],
                        choices=["off", "priority", "equal", "adaptive", "rlmr", "rlmr_v2"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--map-offset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--retransmission-seed", type=int, default=2000)
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument("--output-tag", default="heldout_seed1000")
    parser.add_argument("--q-table", default="mar_inference/rlmr_q_table.json",
                        help="Frozen RLMR-v1 Q table.")
    parser.add_argument("--v2-q-table", default="mar_inference/rlmr_v2_q_table.json",
                        help="Frozen RLMR-v2 Q table.")
    parser.add_argument("--v2-1-q-table", default="mar_inference/rlmr_v2_1_q_table.json",
                        help="Frozen RLMR-v2.1 Q table.")
    parser.add_argument("--v2-profile", choices=["v2", "v2_1"], default="v2")
    parser.add_argument("--meta-agents", type=int, default=None,
                        help="Override Ray worker count for each sweep.")
    parser.add_argument("--no-map-shield", action="store_true",
                        help="Evaluate RLMR-v2 without its critical-map safety constraint.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    q_tables = {}
    if "rlmr" in args.policies:
        q_tables["v1"] = ROOT / args.q_table
    if "rlmr_v2" in args.policies:
        selected_v2_table = args.v2_q_table if args.v2_profile == "v2" else args.v2_1_q_table
        q_tables[args.v2_profile] = ROOT / selected_v2_table
    for version, q_table in q_tables.items():
        if not args.dry_run and not q_table.is_file():
            raise FileNotFoundError(f"RLMR-{version} Q table not found: {q_table}")
    frozen_hashes = {
        version: file_sha256(path)
        for version, path in q_tables.items()
        if path.is_file()
    }
    for version, digest in frozen_hashes.items():
        print(f"Frozen RLMR-{version} Q-table SHA-256: {digest}", flush=True)

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
            "--simulation-seed", str(args.simulation_seed),
            "--output-tag", args.output_tag,
            "--v1-q-table", str(ROOT / args.q_table),
            "--v2-q-table", str(ROOT / args.v2_q_table),
            "--v2-1-q-table", str(ROOT / args.v2_1_q_table),
            "--rlmr-v2-profile", args.v2_profile,
        ]
        if args.meta_agents is not None:
            command.extend(["--meta-agents", str(args.meta_agents)])
        if args.overwrite:
            command.append("--overwrite")
        if policy == "rlmr_v2" and args.no_map_shield:
            command.append("--no-map-shield")

        print("=" * 80, flush=True)
        print(f"Held-out evaluation policy={policy}", flush=True)
        print(" ".join(command), flush=True)
        print("=" * 80, flush=True)

        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            for version, q_table in q_tables.items():
                current_hash = file_sha256(q_table)
                if current_hash != frozen_hashes[version]:
                    raise RuntimeError(
                        f"RLMR-{version} Q table changed during evaluation: "
                        f"expected {frozen_hashes[version]}, got {current_hash}"
                    )
                print(
                    f"RLMR-{version} Q table unchanged after policy={policy}: {current_hash}",
                    flush=True,
                )

    print("Held-out retransmission evaluation completed.", flush=True)


if __name__ == "__main__":
    main()
