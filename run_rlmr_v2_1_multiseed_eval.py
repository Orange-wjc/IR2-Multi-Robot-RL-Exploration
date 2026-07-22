#!/usr/bin/env python3
"""Run resumable multi-seed held-out evaluation for frozen RLMR-v2.1."""

import argparse
import csv
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SWEEP_SCRIPT = ROOT / "run_message_loss_sweep.py"
LOG_DIR = ROOT / "mar_inference" / "test_results" / "log"
FROZEN_HASHES = {
    "v2": "985f1699c1c990b0b712e40e69f3831abf4c3a1db3c5b57d30ff94e5db0d12b4",
    "v2_1": "bb5f5919ec5d001083a300e9d8397b4388d4827f4a5233fb8278adf018978a54",
}
POLICY_CONFIG = {
    "off": ("off", "v2_1", ""),
    "priority": ("priority", "v2_1", "_priority_retrans"),
    "v2": ("rlmr_v2", "v2", "_rlmr_v2_retrans"),
    "v2_1": ("rlmr_v2", "v2_1", "_rlmr_v2_1_retrans"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loss_suffix(loss):
    return f"{int(round(loss * 10)):02d}"


def output_tag(prefix, message_seed_base):
    return f"{prefix}_seed{message_seed_base}"


def result_paths(map_name, loss, policy, tag, log_dir=LOG_DIR):
    retrans_suffix = POLICY_CONFIG[policy][2]
    stem = f"{map_name}_msg_random_{loss_suffix(loss)}{retrans_suffix}_{tag}"
    return log_dir / f"{stem}.csv", log_dir / f"{stem}_skipped.csv"


def build_command(args, policy, loss, message_seed, retransmission_seed, tag):
    retransmission, profile, _ = POLICY_CONFIG[policy]
    command = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--maps", args.map,
        "--modes", "random",
        "--losses", str(loss),
        "--retransmission", retransmission,
        "--episodes", str(args.episodes),
        "--map-offset", str(args.map_offset),
        "--seed", str(message_seed),
        "--retransmission-seed", str(retransmission_seed),
        "--simulation-seed", str(args.simulation_seed),
        "--output-tag", tag,
        "--v2-q-table", str(args.v2_q_table),
        "--v2-1-q-table", str(args.v2_1_q_table),
        "--rlmr-v2-profile", profile,
        "--meta-agents", str(args.meta_agents),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_result(main_path, skipped_path, args, policy, loss, message_seed, retransmission_seed):
    if not main_path.is_file() or not skipped_path.is_file():
        raise FileNotFoundError(f"Missing result pair: {main_path.name}, {skipped_path.name}")

    main_rows = read_rows(main_path)
    skipped_rows = read_rows(skipped_path)
    records = [("valid", row) for row in main_rows] + [("skipped", row) for row in skipped_rows]
    if len(records) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} attempted episodes for {main_path.name}, got {len(records)}"
        )

    map_indices = [int(row["map_index"]) for _, row in records]
    expected_maps = set(range(args.map_offset, args.map_offset + args.episodes))
    if len(set(map_indices)) != len(map_indices) or set(map_indices) != expected_maps:
        raise RuntimeError(f"Map coverage mismatch for {main_path.name}")

    expected_version = "v2_1" if policy == "v2_1" else "v2" if policy == "v2" else "off"
    for kind, row in records:
        map_index = int(row["map_index"])
        if int(row["message_loss_seed"]) != message_seed:
            raise RuntimeError(f"Message seed mismatch for map {map_index}")
        if int(row["retransmission_loss_seed"]) != retransmission_seed:
            raise RuntimeError(f"Retransmission seed mismatch for map {map_index}")
        if int(row["simulation_seed"]) != args.simulation_seed + map_index:
            raise RuntimeError(f"Simulation seed mismatch for map {map_index}")
        if not math.isclose(float(row["message_loss_prob"]), loss):
            raise RuntimeError(f"Message loss mismatch for map {map_index}")
        if row.get("rlmr_version", "off") != expected_version:
            raise RuntimeError(f"RLMR version mismatch for map {map_index}")
        if kind == "skipped":
            required = (
                "retrans_attempts",
                "map_retrans_attempts",
                "pose_retrans_attempts",
                "graph_retrans_attempts",
            )
            if any(name not in row for name in required):
                raise RuntimeError(f"Skipped communication columns missing for map {map_index}")
            parts = sum(int(float(row[name] or 0)) for name in required[1:])
            aggregate = int(float(row[required[0]] or 0))
            if aggregate != parts:
                raise RuntimeError(f"Skipped retransmission total mismatch for map {map_index}")

    return len(main_rows), len(skipped_rows)


def selected_q_tables(args):
    tables = {}
    if "v2" in args.policies:
        tables["v2"] = Path(args.v2_q_table)
    if "v2_1" in args.policies:
        tables["v2_1"] = Path(args.v2_1_q_table)
    return {
        version: path if path.is_absolute() else ROOT / path
        for version, path in tables.items()
    }


def verify_frozen_tables(q_tables, dry_run=False):
    hashes = {}
    for version, path in q_tables.items():
        if not path.is_file():
            if dry_run:
                continue
            raise FileNotFoundError(f"Frozen RLMR-{version} Q table not found: {path}")
        digest = file_sha256(path)
        expected = FROZEN_HASHES[version]
        if digest != expected:
            raise RuntimeError(
                f"Unexpected RLMR-{version} Q-table hash: expected {expected}, got {digest}"
            )
        hashes[version] = digest
    return hashes


def stop_ray(dry_run=False):
    command = ["ray", "stop", "--force"]
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable multi-seed Priority vs frozen RLMR-v2.1 evaluation."
    )
    parser.add_argument("--map", default="corridor", choices=["corridor", "hybrid", "complex"])
    parser.add_argument("--losses", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--policies", nargs="+", choices=POLICY_CONFIG, default=["priority", "v2_1"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--map-offset", type=int, default=200)
    parser.add_argument("--simulation-seed", type=int, default=7000)
    parser.add_argument("--message-seed-bases", nargs="+", type=int, default=[15000, 25000])
    parser.add_argument("--retransmission-seed-bases", nargs="+", type=int, default=[16000, 26000])
    parser.add_argument("--meta-agents", type=int, default=1)
    parser.add_argument("--output-prefix", default="v2_1_multiseed")
    parser.add_argument("--v2-q-table", default="mar_inference/rlmr_v2_q_table.json")
    parser.add_argument("--v2-1-q-table", default="mar_inference/rlmr_v2_1_q_table.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0 or args.map_offset < 0 or args.meta_agents <= 0:
        raise ValueError("episodes and meta-agents must be positive; map-offset must be non-negative")
    if any(loss <= 0 or loss > 1 for loss in args.losses):
        raise ValueError("losses must be in (0, 1]")
    if len(args.message_seed_bases) != len(args.retransmission_seed_bases):
        raise ValueError("message and retransmission seed base lists must have equal length")
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("policies must not contain duplicates")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_prefix):
        raise ValueError("output-prefix may contain only letters, numbers, underscores, and hyphens")


def main():
    args = parse_args()
    validate_args(args)
    q_tables = selected_q_tables(args)
    frozen_hashes = verify_frozen_tables(q_tables, args.dry_run)
    for version, digest in frozen_hashes.items():
        print(f"Frozen RLMR-{version} Q-table SHA-256: {digest}", flush=True)

    completed = 0
    reused = 0
    planned = 0
    total = len(args.message_seed_bases) * len(args.losses) * len(args.policies)
    for message_base, retransmission_base in zip(
        args.message_seed_bases,
        args.retransmission_seed_bases,
    ):
        tag = output_tag(args.output_prefix, message_base)
        for loss_index, loss in enumerate(args.losses):
            message_seed = message_base + loss_index
            retransmission_seed = retransmission_base + loss_index
            for policy in args.policies:
                main_path, skipped_path = result_paths(args.map, loss, policy, tag)
                if main_path.exists() and skipped_path.exists() and not args.overwrite:
                    valid, skipped = validate_result(
                        main_path,
                        skipped_path,
                        args,
                        policy,
                        loss,
                        message_seed,
                        retransmission_seed,
                    )
                    reused += 1
                    print(
                        f"Reusing policy={policy}, loss={loss:.1f}, seed={message_seed}: "
                        f"valid={valid}, skipped={skipped}",
                        flush=True,
                    )
                    continue
                if main_path.exists() != skipped_path.exists() and not args.overwrite:
                    raise RuntimeError(
                        f"Incomplete result pair for {main_path.name}; use --overwrite to replace it"
                    )

                stop_ray(args.dry_run)
                command = build_command(
                    args,
                    policy,
                    loss,
                    message_seed,
                    retransmission_seed,
                    tag,
                )
                print("=" * 80, flush=True)
                print(
                    f"Multi-seed run {completed + reused + planned + 1}/{total}: policy={policy}, "
                    f"loss={loss:.1f}, message_seed={message_seed}, "
                    f"retransmission_seed={retransmission_seed}",
                    flush=True,
                )
                print(" ".join(command), flush=True)
                print("=" * 80, flush=True)
                if args.dry_run:
                    planned += 1
                    continue

                subprocess.run(command, cwd=ROOT, check=True)
                valid, skipped = validate_result(
                    main_path,
                    skipped_path,
                    args,
                    policy,
                    loss,
                    message_seed,
                    retransmission_seed,
                )
                current_hashes = verify_frozen_tables(q_tables)
                if current_hashes != frozen_hashes:
                    raise RuntimeError("A frozen Q table changed during multi-seed evaluation")
                completed += 1
                print(f"Validated valid={valid}, skipped={skipped}", flush=True)

    stop_ray(args.dry_run)
    if args.dry_run:
        print(f"Multi-seed dry run complete: total={total}, planned={planned}, reused={reused}", flush=True)
    else:
        print(
            f"Multi-seed evaluation complete: total={total}, executed={completed}, reused={reused}",
            flush=True,
        )


if __name__ == "__main__":
    main()
