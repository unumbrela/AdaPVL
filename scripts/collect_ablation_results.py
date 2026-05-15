#!/usr/bin/env python
"""
Collect Kvasir 10% ablation results for AdaPVL.

Usage:
    python scripts/collect_ablation_results.py --output-dir output_ablation --seed 1
"""
import argparse
import glob
import os
import pandas as pd


RUN_NAMES = [
    "aagf_only",
    "aagf_cmas",
    "full",
    "global_gates",
    "no_cmas",
    "all10_layers",
]

BACKBONES = [
    "adapvl_evaclip",
    "adapvl_dinov3",
]


def find_csv(output_dir: str, run_name: str, seed: int, backbone: str):
    pattern = os.path.join(
        output_dir,
        run_name,
        "Kvasir_10",
        "seg_results",
        f"seed{seed}",
        f"test_MedCLIPSeg_{backbone}_*_Prompt-original.csv",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for run_name in RUN_NAMES:
        row = {"Configuration": run_name}
        for backbone in BACKBONES:
            csv_path = find_csv(args.output_dir, run_name, args.seed, backbone)
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                row[backbone] = round(df["DSC"].mean() * 100, 2)
            else:
                row[backbone] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
