#!/usr/bin/env python3
"""Run two identical held-out evaluations and compare every episode."""

import argparse
import csv
import hashlib
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "mar_inference" / "test_results" / "log"
EVAL_SCRIPT = ROOT / "run_heldout_retransmission_eval.py"
IGNORED_FIELDS = {"meta_agent_id"}
POLICY_STEMS = {
    "priority": "priority_retrans",
    "v2_1": "rlmr_v2_1_retrans",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loss_suffix(loss):
    return f"{int(round(loss * 10)):02d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify deterministic Priority and frozen RLMR-v2.1 inference."
    )
    parser.add_argument("--map", default="corridor", choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--loss", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--map-offset", type=int, default=200)
    parser.add_argument("--message-seed", type=int, default=5000)
    parser.add_argument("--retransmission-seed", type=int, default=6000)
    parser.add_argument("--simulation-seed", type=int, default=7000)
    parser.add_argument("--meta-agents", type=int, default=2)
    parser.add_argument("--q-table", default="mar_inference/rlmr_v2_1_q_table.json")
    parser.add_argument("--output-tag", default="v2_1_repro")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, repeat):
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--map", args.map,
        "--losses", str(args.loss),
        "--policies", "priority", "rlmr_v2",
        "--episodes", str(args.episodes),
        "--map-offset", str(args.map_offset),
        "--seed", str(args.message_seed),
        "--retransmission-seed", str(args.retransmission_seed),
        "--simulation-seed", str(args.simulation_seed),
        "--output-tag", f"{args.output_tag}_{repeat}",
        "--v2-profile", "v2_1",
        "--v2-1-q-table", args.q_table,
        "--meta-agents", str(args.meta_agents),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def result_paths(args, policy, repeat):
    stem = POLICY_STEMS[policy]
    tag = f"{args.output_tag}_{repeat}"
    base = f"{args.map}_msg_random_{loss_suffix(args.loss)}_{stem}_{tag}"
    return LOG_DIR / f"{base}.csv", LOG_DIR / f"{base}_skipped.csv"


def load_episode_records(main_path, skipped_path):
    records = {}
    for record_type, path in (("completed", main_path), ("skipped", skipped_path)):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                episode = int(row["eps"])
                if episode in records:
                    raise RuntimeError(f"Episode {episode} appears more than once in {path}")
                normalized = {
                    key: value
                    for key, value in row.items()
                    if key not in IGNORED_FIELDS
                }
                records[episode] = (record_type, normalized)
    return records


def compare_policy(args, policy):
    first = load_episode_records(*result_paths(args, policy, "a"))
    second = load_episode_records(*result_paths(args, policy, "b"))
    expected = set(range(args.episodes))
    if set(first) != expected or set(second) != expected:
        raise RuntimeError(
            f"{policy} did not produce exactly episodes 0-{args.episodes - 1}: "
            f"repeat_a={sorted(first)}, repeat_b={sorted(second)}"
        )

    mismatches = []
    for episode in sorted(expected):
        if first[episode] == second[episode]:
            continue
        first_type, first_row = first[episode]
        second_type, second_row = second[episode]
        changed_fields = sorted(
            key
            for key in set(first_row) | set(second_row)
            if first_row.get(key) != second_row.get(key)
        )
        mismatches.append((episode, first_type, second_type, changed_fields))

    if mismatches:
        details = "; ".join(
            f"eps={episode} {first_type}->{second_type} fields={fields[:8]}"
            for episode, first_type, second_type, fields in mismatches[:10]
        )
        raise RuntimeError(f"{policy} reproducibility check failed: {details}")
    print(f"PASS: {policy} produced {args.episodes} identical episode records.", flush=True)


def validate_args(args):
    if args.episodes <= 0 or args.meta_agents <= 0:
        raise ValueError("--episodes and --meta-agents must be positive")
    if args.map_offset < 0:
        raise ValueError("--map-offset must be non-negative")
    if args.loss <= 0 or args.loss > 1:
        raise ValueError("--loss must be in (0, 1]")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        raise ValueError("--output-tag may contain only letters, numbers, underscores, and hyphens")


def main():
    args = parse_args()
    validate_args(args)
    q_table = Path(args.q_table)
    if not q_table.is_absolute():
        q_table = ROOT / q_table
    if not args.dry_run and not q_table.is_file():
        raise FileNotFoundError(f"RLMR-v2.1 Q table not found: {q_table}")
    frozen_hash = file_sha256(q_table) if q_table.is_file() else None

    for repeat in ("a", "b"):
        command = build_command(args, repeat)
        print("=" * 80, flush=True)
        print(f"Reproducibility repeat {repeat.upper()}", flush=True)
        print(shlex.join(command), flush=True)
        print("=" * 80, flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            current_hash = file_sha256(q_table)
            if current_hash != frozen_hash:
                raise RuntimeError(
                    f"RLMR-v2.1 Q table changed: expected {frozen_hash}, got {current_hash}"
                )

    if not args.dry_run:
        compare_policy(args, "priority")
        compare_policy(args, "v2_1")
        print(f"Frozen Q-table SHA-256: {frozen_hash}", flush=True)
    print("Reproducibility check completed.", flush=True)


if __name__ == "__main__":
    main()
