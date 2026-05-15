#!/usr/bin/env python
"""
Run the remaining main benchmark first, then continue ablations.

This script is intended to be launched in the background from the llm conda
environment so the same Python interpreter is reused for both stages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="/tmp/continue_main_then_ablation.log")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--main-output-dir", default="output_adapvl_dinov3_siglip_full")
    parser.add_argument("--ablation-output-dir", default="output_ablation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("ab", buffering=0) as log:
        header = f"\n=== restart {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        log.write(header.encode())

        subprocess.run(
            [
                sys.executable,
                "scripts/continue_adapvl.py",
                "--output-dir",
                args.main_output_dir,
                "--seed",
                str(args.seed),
                "--clip-model",
                "adapvl_dinov3_siglip",
                "--train-batch-size",
                str(args.train_batch_size),
            ],
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/continue_ablation.py",
                "--output-dir",
                args.ablation_output_dir,
                "--seed",
                str(args.seed),
                "--train-batch-size",
                str(args.train_batch_size),
            ],
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=True,
        )


if __name__ == "__main__":
    main()
