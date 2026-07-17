#!/usr/bin/env python3
"""Run the two-stage RLMR-v2 development evaluation."""

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = ROOT / "run_heldout_retransmission_eval.py"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Priority vs RLMR-v2, then the no-shield RLMR-v2 ablation."
    )
    parser.add_argument("--map", default="corridor", choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--map-offset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--retransmission-seed", type=int, default=4000)
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument("--output-tag", default=None,
                        help="Output suffix. Defaults to a profile-specific development tag.")
    parser.add_argument("--profile", choices=["v2", "v2_1"], default="v2")
    parser.add_argument("--v2-q-table", default="mar_inference/rlmr_v2_q_table.json")
    parser.add_argument("--v2-1-q-table", default="mar_inference/rlmr_v2_1_q_table.json")
    parser.add_argument("--meta-agents", type=int, default=2,
                        help="Ray workers per setting. Default: 2 to reduce OOM risk.")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["comparison", "no_shield"],
        default=["comparison", "no_shield"],
        help="Phases to run. Default: comparison no_shield.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, phase):
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--map", args.map,
        "--losses", *[str(loss) for loss in args.losses],
        "--episodes", str(args.episodes),
        "--map-offset", str(args.map_offset),
        "--seed", str(args.seed),
        "--retransmission-seed", str(args.retransmission_seed),
        "--simulation-seed", str(args.simulation_seed),
        "--output-tag", args.output_tag,
        "--v2-q-table", str(Path(args.v2_q_table)),
        "--v2-1-q-table", str(Path(args.v2_1_q_table)),
        "--v2-profile", args.profile,
        "--meta-agents", str(args.meta_agents),
    ]
    if phase == "comparison":
        command.extend(["--policies", "priority", "rlmr_v2"])
    else:
        command.extend(["--policies", "rlmr_v2", "--no-map-shield"])
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.map_offset < 0:
        raise ValueError("--map-offset must be non-negative")
    if args.meta_agents <= 0:
        raise ValueError("--meta-agents must be positive")
    if any(loss <= 0 or loss > 1 for loss in args.losses):
        raise ValueError("--losses values must be in (0, 1]")

    if args.output_tag is None:
        args.output_tag = f"{args.profile}_dev{args.episodes}_seed{args.seed}"

    selected_q_table = args.v2_q_table if args.profile == "v2" else args.v2_1_q_table
    q_table = Path(selected_q_table)
    if not q_table.is_absolute():
        q_table = ROOT / q_table
    if not args.dry_run and not q_table.is_file():
        raise FileNotFoundError(f"RLMR-v2 Q table not found: {q_table}")

    frozen_hash = file_sha256(q_table) if q_table.is_file() else None
    if frozen_hash is not None:
        print(f"Frozen RLMR-{args.profile} Q-table SHA-256: {frozen_hash}", flush=True)
    else:
        print(f"Dry run: RLMR-{args.profile} Q table not checked: {q_table}", flush=True)

    for index, phase in enumerate(args.phases, start=1):
        command = build_command(args, phase)
        print("=" * 80, flush=True)
        print(f"RLMR-v2 development phase {index}/{len(args.phases)}: {phase}", flush=True)
        print(shlex.join(command), flush=True)
        print("=" * 80, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

        if not args.dry_run:
            current_hash = file_sha256(q_table)
            if current_hash != frozen_hash:
                raise RuntimeError(
                    f"RLMR-{args.profile} Q table changed during development evaluation: "
                    f"expected {frozen_hash}, got {current_hash}"
                )
            print(f"Q table unchanged after phase={phase}: {current_hash}", flush=True)

    total_runs = len(args.losses) * args.episodes
    comparison_runs = 2 * total_runs if "comparison" in args.phases else 0
    no_shield_runs = total_runs if "no_shield" in args.phases else 0
    print("RLMR-v2 development evaluation completed.", flush=True)
    print(
        f"Attempted episodes scheduled: {comparison_runs + no_shield_runs} "
        f"(comparison={comparison_runs}, no_shield={no_shield_runs})",
        flush=True,
    )


if __name__ == "__main__":
    main()
