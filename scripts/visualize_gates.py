"""
Visualize learned AdaPVL gate values across layers.

Usage:
    python scripts/visualize_gates.py \
        --checkpoint output_adapvl/Kvasir_10/trained_models/seed1/MedCLIPSeg_adapvl_evaclip_ViT-B-16_best_dice.pth \
        --output-path papers/figures/gate_patterns.png

This produces the key interpretability figure for the paper:
    - X-axis: fusion layer (1-10)
    - Y-axis: sigmoid(gate) value
    - Two lines: vision gate (solid) and text gate (dashed)
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


def extract_gates(checkpoint_path):
    """Extract gate values from a trained AdaPVL checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    gate_vis_values = []
    gate_txt_values = []

    for i in range(10):  # 10 fusion layers
        key_vis = f'pvl_adapters.{i}.gate_vis'
        key_txt = f'pvl_adapters.{i}.gate_txt'
        if key_vis in state_dict:
            gate_vis_values.append(torch.sigmoid(state_dict[key_vis]).item())
            gate_txt_values.append(torch.sigmoid(state_dict[key_txt]).item())

    return gate_vis_values, gate_txt_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='Path(s) to checkpoint file(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Label(s) for each checkpoint')
    parser.add_argument('--output-path', default='gate_patterns.png')
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#795548']
    layers = list(range(1, 11))

    labels = args.labels or [f'Model {i+1}' for i in range(len(args.checkpoint))]
    if len(labels) != len(args.checkpoint):
        raise ValueError('--labels must have the same number of entries as --checkpoint')

    for idx, (ckpt_path, label) in enumerate(zip(args.checkpoint, labels)):
        gv, gt = extract_gates(ckpt_path)
        color = colors[idx % len(colors)]

        axes[0].plot(layers, gv, 'o-', color=color, label=label, linewidth=2, markersize=5)
        axes[1].plot(layers, gt, 's--', color=color, label=label, linewidth=2, markersize=5)

    for ax, title in zip(axes, ['Vision Gate $\\sigma(g_v^l)$',
                                 'Text Gate $\\sigma(g_t^l)$']):
        ax.set_xlabel('Fusion Layer', fontsize=12)
        ax.set_ylabel('Gate Value', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Init (0.5)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    plt.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved gate visualization to {output_path}")

    # Also print numerical values
    for ckpt_path, label in zip(args.checkpoint, labels):
        gv, gt = extract_gates(ckpt_path)
        print(f"\n{label}:")
        print(f"  Vision gates: {[f'{v:.3f}' for v in gv]}")
        print(f"  Text gates:   {[f'{v:.3f}' for v in gt]}")


if __name__ == '__main__':
    main()
