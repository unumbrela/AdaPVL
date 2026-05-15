#!/usr/bin/env python
"""Compare segmentation results across all model variants."""
import os
import argparse
import pandas as pd
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--dataset", type=str, default="Kvasir")
    parser.add_argument("--data_percentage", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--prompt_design", type=str, default="original")
    args = parser.parse_args()

    dataset_name = f"{args.dataset}_{args.data_percentage}" if args.data_percentage != 100 else args.dataset
    results_dir = os.path.join(args.output_dir, dataset_name, "seg_results", f"seed{args.seed}")

    # Find all test CSV files
    pattern = os.path.join(results_dir, f"test_MedCLIPSeg_*_Prompt-{args.prompt_design}.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        # Fallback: try the old naming convention
        old_csv = os.path.join(results_dir, f"test_Prompt-{args.prompt_design}.csv")
        if os.path.exists(old_csv):
            csv_files = [old_csv]

    if not csv_files:
        print(f"No result CSVs found in {results_dir}")
        return

    rows = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        # Extract model name from filename like "test_MedCLIPSeg_evaclip_ViT-B-16_Prompt-original.csv"
        parts = fname.replace("test_MedCLIPSeg_", "").replace(f"_Prompt-{args.prompt_design}.csv", "")
        model_name = parts

        df = pd.read_csv(csv_path)
        rows.append({
            "Model": model_name,
            "DSC (%)": round(df["DSC"].mean() * 100, 2),
            "NSD (%)": round(df["NSD"].mean() * 100, 2),
            "Num Samples": len(df),
        })

    summary = pd.DataFrame(rows).sort_values("DSC (%)", ascending=False)

    print("\n" + "=" * 60)
    print(f"Comparison: {dataset_name} | seed={args.seed} | prompt={args.prompt_design}")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    # Save comparison table
    out_path = os.path.join(results_dir, "model_comparison.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
