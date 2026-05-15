#!/usr/bin/env python
"""
Collect and summarize all experiment results for comparison with the original paper.
Computes per-dataset DSC/NSD and overall averages for both Data Efficiency
and Domain Generalization evaluations.

Usage:
    python scripts/collect_results.py --output-dir output_evaclip --seed 1
    python scripts/collect_results.py --output-dir output_evaclip --seed 1 --clip-model evaclip
"""
import os
import argparse
import pandas as pd
import glob


# Original paper results (MedCLIPSeg with UniMedCLIP, seed-averaged)
PAPER_RESULTS_EFFICIENCY = {
    # Average DSC across 6 datasets at each data percentage
    10: 81.10,
    25: 85.08,
    50: 87.18,
    100: 88.66,
}

PAPER_RESULTS_DG = {
    "ID": 89.11,
    "OOD": 79.02,
    "HM": 83.76,
}

# Note: EUS may be missing (no image data) — will show as N/A
EFFICIENCY_DATASETS = ["BUSI", "BTMRI", "ISIC", "Kvasir", "Covid19", "EUS"]
DATA_PERCENTAGES = [10, 25, 50, 100]

DOMAINS = {
    "BUS": ["BUSI", "BUSBRA", "BUSUC", "BUID", "UDIAT"],
    "ENDO": ["Kvasir", "ColonDB", "ClinicDB", "CVC300", "BKAI"],
    "DERM": ["ISIC", "UWaterlooSkinCancer"],
    "BRAIN": ["BTMRI", "BRISC"],
}


def find_eval_csv(output_dir, dataset, data_pct, seed, clip_model):
    """Find the evaluation CSV for a given experiment."""
    ds_name = f"{dataset}_{data_pct}" if data_pct != 100 else dataset
    results_dir = os.path.join(output_dir, ds_name, "seg_results", f"seed{seed}")

    # Try new naming: test_MedCLIPSeg_<clip_model>_*_Prompt-original.csv
    pattern = os.path.join(results_dir, f"test_MedCLIPSeg_{clip_model}_*_Prompt-original.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Fallback: old naming
    old_path = os.path.join(results_dir, "test_Prompt-original.csv")
    if os.path.exists(old_path):
        return old_path

    return None


def load_metrics(csv_path):
    """Load DSC and NSD from eval CSV."""
    df = pd.read_csv(csv_path)
    return df["DSC"].mean() * 100, df["NSD"].mean() * 100


def collect_efficiency(args):
    """Collect data efficiency results."""
    print("\n" + "=" * 80)
    print(" DATA EFFICIENCY EVALUATION")
    print("=" * 80)

    rows = []
    for dataset in EFFICIENCY_DATASETS:
        row = {"Dataset": dataset}
        for pct in DATA_PERCENTAGES:
            csv_path = find_eval_csv(args.output_dir, dataset, pct, args.seed, args.clip_model)
            if csv_path and os.path.exists(csv_path):
                dsc, nsd = load_metrics(csv_path)
                row[f"DSC_{pct}%"] = round(dsc, 2)
                row[f"NSD_{pct}%"] = round(nsd, 2)
            else:
                row[f"DSC_{pct}%"] = None
                row[f"NSD_{pct}%"] = None
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print per-dataset results
    print(f"\nPer-dataset DSC (%) | clip_model={args.clip_model} | seed={args.seed}")
    print("-" * 80)
    dsc_cols = [f"DSC_{p}%" for p in DATA_PERCENTAGES]
    print(df[["Dataset"] + dsc_cols].to_string(index=False))

    # Compute averages
    print("\n" + "-" * 80)
    print("AVERAGE DSC across 6 datasets:")
    avg_row = {"": "EVA02-CLIP (Ours)"}
    paper_row = {"": "MedCLIPSeg (Paper)"}
    diff_row = {"": "Improvement"}

    for pct in DATA_PERCENTAGES:
        col = f"DSC_{pct}%"
        valid = df[col].dropna()
        if len(valid) > 0:
            avg = valid.mean()
            paper_val = PAPER_RESULTS_EFFICIENCY[pct]
            n = len(valid)
            avg_row[f"{pct}%"] = f"{avg:.2f} (n={n})"
            paper_row[f"{pct}%"] = f"{paper_val:.2f} (n=6)"
            diff_row[f"{pct}%"] = f"{avg - paper_val:+.2f}"
        else:
            avg_row[f"{pct}%"] = "N/A"
            paper_row[f"{pct}%"] = f"{PAPER_RESULTS_EFFICIENCY[pct]:.2f}"
            diff_row[f"{pct}%"] = "N/A"

    summary = pd.DataFrame([paper_row, avg_row, diff_row])
    print(summary.to_string(index=False))

    return df


def collect_domain_gen(args):
    """Collect domain generalization results."""
    print("\n" + "=" * 80)
    print(" DOMAIN GENERALIZATION EVALUATION")
    print("=" * 80)

    all_id_dsc = []
    all_ood_dsc = []
    rows = []

    for domain, datasets in DOMAINS.items():
        source = datasets[0]
        targets = datasets[1:]

        # In-domain: source trained on 100% and tested on source
        csv_path = find_eval_csv(args.output_dir, source, 100, args.seed, args.clip_model)
        if csv_path and os.path.exists(csv_path):
            id_dsc, id_nsd = load_metrics(csv_path)
            all_id_dsc.append(id_dsc)
            rows.append({
                "Domain": domain, "Type": "ID", "Dataset": source,
                "DSC": round(id_dsc, 2), "NSD": round(id_nsd, 2)
            })
        else:
            rows.append({
                "Domain": domain, "Type": "ID", "Dataset": source,
                "DSC": None, "NSD": None
            })

        # OOD: source model tested on each target
        for target in targets:
            csv_path = find_eval_csv(args.output_dir, target, 100, args.seed, args.clip_model)
            if csv_path and os.path.exists(csv_path):
                ood_dsc, ood_nsd = load_metrics(csv_path)
                all_ood_dsc.append(ood_dsc)
                rows.append({
                    "Domain": domain, "Type": "OOD", "Dataset": target,
                    "DSC": round(ood_dsc, 2), "NSD": round(ood_nsd, 2)
                })
            else:
                rows.append({
                    "Domain": domain, "Type": "OOD", "Dataset": target,
                    "DSC": None, "NSD": None
                })

    df = pd.DataFrame(rows)
    print(f"\nPer-dataset DSC (%) | clip_model={args.clip_model} | seed={args.seed}")
    print("-" * 80)
    print(df.to_string(index=False))

    # Compute averages
    if all_id_dsc and all_ood_dsc:
        avg_id = sum(all_id_dsc) / len(all_id_dsc)
        avg_ood = sum(all_ood_dsc) / len(all_ood_dsc)
        hm = 2 * avg_id * avg_ood / (avg_id + avg_ood) if (avg_id + avg_ood) > 0 else 0

        print("\n" + "-" * 80)
        print("SUMMARY:")
        print(f"{'':30s} {'ID':>8s} {'OOD':>8s} {'HM':>8s}")
        print(f"{'MedCLIPSeg (Paper)':30s} {PAPER_RESULTS_DG['ID']:8.2f} {PAPER_RESULTS_DG['OOD']:8.2f} {PAPER_RESULTS_DG['HM']:8.2f}")
        print(f"{'EVA02-CLIP (Ours)':30s} {avg_id:8.2f} {avg_ood:8.2f} {hm:8.2f}")
        print(f"{'Improvement':30s} {avg_id - PAPER_RESULTS_DG['ID']:+8.2f} {avg_ood - PAPER_RESULTS_DG['OOD']:+8.2f} {hm - PAPER_RESULTS_DG['HM']:+8.2f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output_evaclip")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clip-model", type=str, default="evaclip")
    args = parser.parse_args()

    eff_df = collect_efficiency(args)
    dg_df = collect_domain_gen(args)

    # Save to CSV
    out_path = os.path.join(args.output_dir, f"full_comparison_seed{args.seed}.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# Data Efficiency Results\n")
        eff_df.to_csv(f, index=False)
        f.write("\n# Domain Generalization Results\n")
        dg_df.to_csv(f, index=False)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
