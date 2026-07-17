#!/usr/bin/env python3
"""Train RLMR-v2.1 and run its development evaluation."""

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_rlmr_v2.py"
EVAL_SCRIPT = ROOT / "run_rlmr_v2_dev_eval.py"
DEFAULT_Q_TABLE = "mar_inference/rlmr_v2_1_q_table.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RLMR-v2.1, then compare Priority, shield, and no-shield policies."
    )
    parser.add_argument("--train-map", default="corridor",
                        choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--eval-map", default="corridor",
                        choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--episodes-per-setting", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-map-offset", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--train-retransmission-seed", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-map-offset", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=3000)
    parser.add_argument("--eval-retransmission-seed", type=int, default=4000)
    parser.add_argument("--meta-agents", type=int, default=2)
    parser.add_argument("--q-table", default=DEFAULT_Q_TABLE)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Continue training from an existing V2.1 Q table.")
    parser.add_argument("--overwrite-eval", action="store_true",
                        help="Replace development CSV files with matching names.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_train_command(args):
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--profile", "v2_1",
        "--maps", args.train_map,
        "--losses", *[str(loss) for loss in args.losses],
        "--episodes-per-setting", str(args.episodes_per_setting),
        "--epochs", str(args.epochs),
        "--map-offset", str(args.train_map_offset),
        "--seed", str(args.train_seed),
        "--retransmission-seed", str(args.train_retransmission_seed),
        "--q-table", args.q_table,
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_eval_command(args):
    output_tag = args.output_tag or f"v2_1_dev{args.eval_episodes}_seed{args.eval_seed}"
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--profile", "v2_1",
        "--map", args.eval_map,
        "--losses", *[str(loss) for loss in args.losses],
        "--episodes", str(args.eval_episodes),
        "--map-offset", str(args.eval_map_offset),
        "--seed", str(args.eval_seed),
        "--retransmission-seed", str(args.eval_retransmission_seed),
        "--meta-agents", str(args.meta_agents),
        "--v2-1-q-table", args.q_table,
        "--output-tag", output_tag,
    ]
    if args.overwrite_eval:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def validate_args(args):
    if args.episodes_per_setting <= 0 or args.epochs <= 0 or args.eval_episodes <= 0:
        raise ValueError("Training and evaluation episode counts must be positive")
    if args.train_map_offset < 0 or args.eval_map_offset < 0:
        raise ValueError("Map offsets must be non-negative")
    if args.meta_agents <= 0:
        raise ValueError("--meta-agents must be positive")
    if any(loss <= 0 or loss > 1 for loss in args.losses):
        raise ValueError("--losses values must be in (0, 1]")


def run_stage(title, command, dry_run):
    print("=" * 80, flush=True)
    print(title, flush=True)
    print(shlex.join(command), flush=True)
    print("=" * 80, flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = parse_args()
    validate_args(args)

    q_table = Path(args.q_table)
    if not q_table.is_absolute():
        q_table = ROOT / q_table
    if q_table.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"V2.1 Q table already exists: {q_table}. "
            "Use --resume to continue training without overwriting it."
        )

    run_stage("Stage 1/2: train RLMR-v2.1", build_train_command(args), args.dry_run)

    if not args.dry_run:
        if not q_table.is_file():
            raise FileNotFoundError(f"Training completed but no V2.1 Q table was found: {q_table}")
        digest = file_sha256(q_table)
        print(f"Frozen trained RLMR-v2.1 Q-table SHA-256: {digest}", flush=True)

    run_stage(
        "Stage 2/2: run RLMR-v2.1 development comparison",
        build_eval_command(args),
        args.dry_run,
    )
    print("RLMR-v2.1 training and development comparison completed.", flush=True)


if __name__ == "__main__":
    main()
