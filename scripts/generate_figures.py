"""
Figure Generator Script for QSED.

Generates workflow diagrams, architecture visualizations, and illustrative figures.
Outputs saved in docs/figures/ and docs/workflow/.

Paper Reference: Figures 2, 3, 11, 20.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classical.sobel_masks import get_eight_sobel_masks
from src.utils.visualization import plot_eight_directional_gradients
from src.classical.sobel_operator import apply_sobel_masks
from src.classical.image_loader import generate_synthetic_image


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    figures_dir = os.path.join(base_dir, "docs", "figures")
    workflow_dir = os.path.join(base_dir, "docs", "workflow")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(workflow_dir, exist_ok=True)

    print("Generating workflow and architectural figures...")

    # 1. Generate Sobel 8-Mask Visualization (Figure 3)
    masks = get_eight_sobel_masks()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Fig. 3 Eight Masks of Eight-Direction Sobel Algorithm (5x5 Neighborhood)", fontsize=14, fontweight='bold')

    directions = ["0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"]
    labels = ["(a) 0°", "(b) 22.5°", "(c) 45°", "(d) 67.5°", "(e) 90°", "(f) 112.5°", "(g) 135°", "(h) 157.5°"]

    for idx, (d_str, label) in enumerate(zip(directions, labels)):
        r, c = idx // 4, idx % 4
        mask = masks[d_str]
        im = axes[r, c].imshow(mask, cmap='bwr', vmin=-4, vmax=4)
        axes[r, c].set_title(f"{label} Mask", fontsize=12)

        # Annotate matrix numbers inside heatmaps
        for i in range(5):
            for j in range(5):
                val = int(mask[i, j])
                color = "white" if abs(val) >= 3 else "black"
                axes[r, c].text(j, i, str(val), ha="center", va="center", color=color, fontweight='bold')

        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sobel_8_masks_fig3.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Generate 5x5 Neighborhood Window Diagram (Figure 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Fig. 2 5x5 Pixel Neighborhood Window Centered at P(Y, X)", fontsize=14, fontweight='bold')

    grid = np.zeros((5, 5))
    ax.imshow(grid, cmap='Wistia')

    for i in range(5):
        for j in range(5):
            dy = i - 2
            dx = j - 2
            label = f"P(Y{dy:+d}, X{dx:+d})" if (dy != 0 or dx != 0) else "P(Y, X)"
            weight = 'bold' if (dy == 0 and dx == 0) else 'normal'
            ax.text(j, i, label, ha="center", va="center", fontsize=9, fontweight=weight)

    ax.set_xticks(np.arange(-0.5, 5, 1))
    ax.set_yticks(np.arange(-0.5, 5, 1))
    ax.grid(color='black', linestyle='-', linewidth=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(os.path.join(figures_dir, "pixel_neighborhood_5x5_fig2.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Eight Directional Gradients Visualization
    sample_img = generate_synthetic_image(size=64, pattern='checkerboard')
    dir_grads = apply_sobel_masks(sample_img)
    plot_eight_directional_gradients(dir_grads, save_path=os.path.join(figures_dir, "eight_directional_gradient_responses.png"))

    print("Figures successfully generated in docs/figures/!")


if __name__ == "__main__":
    main()
