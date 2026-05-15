#!/usr/bin/env python
"""
Continue Kvasir 10% ablation experiments for AdaPVL.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


RUNS = {
    "aagf_only": ["MODEL.USE_CMAS", "False", "MODEL.USE_MLFA", "False"],
    "aagf_cmas": ["MODEL.USE_CMAS", "True", "MODEL.USE_MLFA", "False"],
    "full": ["MODEL.USE_CMAS", "True", "MODEL.USE_MLFA", "True"],
    "global_gates": [
        "MODEL.USE_CMAS", "True",
        "MODEL.USE_MLFA", "True",
        "MODEL.SHARE_GATES", "True",
    ],
    "no_cmas": ["MODEL.USE_CMAS", "False", "MODEL.USE_MLFA", "True"],
    "all10_layers": [
        "MODEL.USE_CMAS", "True",
        "MODEL.USE_MLFA", "True",
        "MODEL.MLFA_ALL_LAYERS", "True",
    ],
}

BACKBONES = ["adapvl_evaclip", "adapvl_dinov3"]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def csv_exists(output_dir: str, run_name: str, seed: int, clip_model: str) -> bool:
    pattern = os.path.join(
        output_dir,
        run_name,
        "Kvasir_10",
        "seg_results",
        f"seed{seed}",
        f"test_MedCLIPSeg_{clip_model}_*_Prompt-original.csv",
    )
    return bool(glob.glob(pattern))


def latest_ckpt_exists(output_dir: str, run_name: str, seed: int, clip_model: str) -> bool:
    pattern = os.path.join(
        output_dir,
        run_name,
        "Kvasir_10",
        "trained_models",
        f"seed{seed}",
        f"MedCLIPSeg_{clip_model}_*_latest.pth",
    )
    return bool(glob.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=None)
    args = parser.parse_args()

    for run_name, opts in RUNS.items():
        for backbone in BACKBONES:
            if csv_exists(args.output_dir, run_name, args.seed, backbone):
                print(f"Skipping completed ablation: {run_name} | {backbone}", flush=True)
                continue

            base_train = [
                sys.executable,
                "train.py",
                "--config-file",
                "configs/Kvasir.yaml",
                "--seed",
                str(args.seed),
                "--data_percentage",
                "10",
                "--output-dir",
                os.path.join(args.output_dir, run_name),
                "MODEL.CLIP_MODEL",
                backbone,
            ]
            if args.train_batch_size:
                base_train.extend(["TRAIN.BATCH_SIZE", str(args.train_batch_size)])
            train_cmd = base_train + opts

            if latest_ckpt_exists(args.output_dir, run_name, args.seed, backbone):
                run(train_cmd[:1] + ["train.py", "--resume"] + train_cmd[2:])
            else:
                run(train_cmd)

            test_cmd = [
                sys.executable,
                "test.py",
                "--config-file",
                "configs/Kvasir.yaml",
                "--seed",
                str(args.seed),
                "--data_percentage",
                "10",
                "--source_dataset",
                "Kvasir_10",
                "--output-dir",
                os.path.join(args.output_dir, run_name),
                "MODEL.CLIP_MODEL",
                backbone,
                "TEST.USE_LATEST",
                "False",
            ] + opts
            eval_cmd = [
                sys.executable,
                "utils/eval.py",
                "--config-file",
                "configs/Kvasir.yaml",
                "--seed",
                str(args.seed),
                "--data_percentage",
                "10",
                "--output-dir",
                os.path.join(args.output_dir, run_name),
                "MODEL.CLIP_MODEL",
                backbone,
            ]

            run(test_cmd)
            run(eval_cmd)


if __name__ == "__main__":
    main()
