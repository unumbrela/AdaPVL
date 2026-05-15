#!/usr/bin/env python
"""
Continue AdaPVL experiments from partial outputs.

This script skips completed runs, resumes interrupted training from the latest
checkpoint when available, and always tests with the best-dice checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ["BUSI", "BTMRI", "ISIC", "Kvasir", "Covid19", "EUS"]
PERCENTAGES = [10, 25, 50, 100]

DOMAINS = {
    "BUSI": ["BUSBRA", "BUSUC", "BUID", "UDIAT"],
    "Kvasir": ["ColonDB", "ClinicDB", "CVC300", "BKAI"],
    "ISIC": ["UWaterlooSkinCancer"],
    "BTMRI": ["BRISC"],
}


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def dataset_name(dataset: str, pct: int) -> str:
    return f"{dataset}_{pct}" if pct != 100 else dataset


def csv_exists(output_dir: str, dataset: str, seed: int, clip_model: str) -> bool:
    pattern = os.path.join(
        output_dir,
        dataset,
        "seg_results",
        f"seed{seed}",
        f"test_MedCLIPSeg_{clip_model}_*_Prompt-original.csv",
    )
    return bool(glob.glob(pattern))


def latest_ckpt_exists(output_dir: str, dataset: str, seed: int, clip_model: str) -> bool:
    pattern = os.path.join(
        output_dir,
        dataset,
        "trained_models",
        f"seed{seed}",
        f"MedCLIPSeg_{clip_model}_*_latest.pth",
    )
    return bool(glob.glob(pattern))


def ensure_train(args, config_file: str, pct: int) -> None:
    ds_name = dataset_name(Path(config_file).stem, pct)
    base_cmd = [
        sys.executable,
        "train.py",
        "--config-file",
        config_file,
        "--seed",
        str(args.seed),
        "--data_percentage",
        str(pct),
        "--output-dir",
        args.output_dir,
        "MODEL.CLIP_MODEL",
        args.clip_model,
    ]
    if args.train_batch_size:
        base_cmd.extend(["TRAIN.BATCH_SIZE", str(args.train_batch_size)])

    if latest_ckpt_exists(args.output_dir, ds_name, args.seed, args.clip_model):
        run(base_cmd[:1] + ["train.py", "--resume"] + base_cmd[2:])
    else:
        run(base_cmd)


def ensure_test_eval(args, config_file: str, pct: int, source_dataset: str) -> None:
    ds_name = dataset_name(Path(config_file).stem, pct)
    if csv_exists(args.output_dir, ds_name, args.seed, args.clip_model):
        print(f"Skipping completed test/eval: {ds_name}", flush=True)
        return

    test_cmd = [
        sys.executable,
        "test.py",
        "--config-file",
        config_file,
        "--seed",
        str(args.seed),
        "--data_percentage",
        str(pct),
        "--source_dataset",
        source_dataset,
        "--output-dir",
        args.output_dir,
        "MODEL.CLIP_MODEL",
        args.clip_model,
        "TEST.USE_LATEST",
        "False",
    ]
    eval_cmd = [
        sys.executable,
        "utils/eval.py",
        "--config-file",
        config_file,
        "--seed",
        str(args.seed),
        "--data_percentage",
        str(pct),
        "--output-dir",
        args.output_dir,
        "MODEL.CLIP_MODEL",
        args.clip_model,
    ]
    run(test_cmd)
    run(eval_cmd)


def continue_data_efficiency(args) -> None:
    for dataset in DATASETS:
        config_file = f"configs/{dataset}.yaml"
        for pct in PERCENTAGES:
            ds_name = dataset_name(dataset, pct)
            if csv_exists(args.output_dir, ds_name, args.seed, args.clip_model):
                print(f"Skipping completed train/test/eval: {ds_name}", flush=True)
                continue
            ensure_train(args, config_file, pct)
            ensure_test_eval(args, config_file, pct, ds_name)


def continue_domain_generalization(args) -> None:
    for source_dataset, targets in DOMAINS.items():
        for target in targets:
            if csv_exists(args.output_dir, target, args.seed, args.clip_model):
                print(f"Skipping completed DG eval: {source_dataset} -> {target}", flush=True)
                continue

            config_file = f"configs/{target}.yaml"
            test_cmd = [
                sys.executable,
                "test.py",
                "--config-file",
                config_file,
                "--seed",
                str(args.seed),
                "--source_dataset",
                source_dataset,
                "--output-dir",
                args.output_dir,
                "MODEL.CLIP_MODEL",
                args.clip_model,
                "TEST.USE_LATEST",
                "False",
            ]
            eval_cmd = [
                sys.executable,
                "utils/eval.py",
                "--config-file",
                config_file,
                "--seed",
                str(args.seed),
                "--output-dir",
                args.output_dir,
                "MODEL.CLIP_MODEL",
                args.clip_model,
            ]
            run(test_cmd)
            run(eval_cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument(
        "--phase",
        choices=["all", "efficiency", "dg"],
        default="all",
    )
    args = parser.parse_args()

    if args.phase in {"all", "efficiency"}:
        continue_data_efficiency(args)
    if args.phase in {"all", "dg"}:
        continue_domain_generalization(args)


if __name__ == "__main__":
    main()
