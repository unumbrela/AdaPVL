"""
Visualize learned MLFA weights across AdaPVL checkpoints.

Usage:
    python scripts/visualize_mlfa.py \
        --checkpoint ckpt1 ckpt2 \
        --labels "AdaPVL+EVA02" "AdaPVL+DINOv3" \
        --output-path papers/figures/mlfa_weights.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_weights(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logits = ckpt["model"]["mlfa.importance_logits"]
    weights = torch.softmax(logits, dim=0).cpu().numpy()
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-path", default="mlfa_weights.png")
    args = parser.parse_args()

    if len(args.checkpoint) != len(args.labels):
        raise ValueError("--checkpoint and --labels must have the same length")

    num_models = len(args.checkpoint)
    cols = 3
    rows = int(np.ceil(num_models / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    layer_labels = ["Layer 3", "Layer 6", "Layer 9"]

    all_weights = []
    for ax, ckpt_path, label in zip(axes.flat, args.checkpoint, args.labels):
        weights = extract_weights(ckpt_path)
        all_weights.append((label, weights))
        ax.bar(layer_labels, weights, color=colors, width=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Weight")
        ax.grid(axis="y", alpha=0.25)
        for idx, weight in enumerate(weights):
            ax.text(idx, weight + 0.02, f"{weight:.2f}", ha="center", va="bottom", fontsize=9)

    for ax in axes.flat[num_models:]:
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved MLFA visualization to {output_path}")

    for label, weights in all_weights:
        formatted = ", ".join(f"{w:.3f}" for w in weights)
        print(f"{label}: [{formatted}]")


if __name__ == "__main__":
    main()
