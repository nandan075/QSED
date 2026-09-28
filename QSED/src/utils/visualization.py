"""
Visualization Module for Quantum Image Edge Detection (QSED).

Generates high-resolution figures displaying the 6 stages of QSED pipeline, 8-directional gradient images,
and comparative plots against classical 2-direction and 4-direction Sobel algorithms.

Paper Reference:
- Section 4.2: Figures 3, 11, 20.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_qsed_pipeline_results(
    pipeline_results: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Generate a 3x3 figure displaying the main progression of the QSED algorithm:
    1. Original Image
    2. Gradient Magnitude G
    3. NMS Image G_S
    4. Double Threshold Map E
    5. Final Edge Detection Output B
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Quantum Image Edge Detection (QSED) — Stage-by-Stage Results", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(pipeline_results['original'], cmap='gray')
    axes[0, 0].set_title("1. Original Image |I>", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pipeline_results['gradient_magnitude'], cmap='magma')
    axes[0, 1].set_title("2. Gradient Magnitude |G>", fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pipeline_results['nms_image'], cmap='magma')
    axes[0, 2].set_title("3. Non-Maximum Suppression |G_S>", fontsize=12)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(pipeline_results['threshold_map'], cmap='plasma')
    axes[1, 0].set_title("4. Double Threshold Map |E>", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pipeline_results['final_edges'], cmap='gray')
    axes[1, 1].set_title("5. Final Quantum Edges |B>", fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pipeline_results['dominant_direction'], cmap='hsv')
    axes[1, 2].set_title("6. Dominant Direction Map (0-7)", fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_eight_directional_gradients(
    directional_gradients: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot the 8 directional gradient images G_0, G_22.5, ..., G_157.5 corresponding to Figure 3.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Eight-Directional Sobel Response Images", fontsize=16, fontweight='bold')

    directions = ["0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"]

    for idx, d_str in enumerate(directions):
        r, c = idx // 4, idx % 4
        grad = np.abs(directional_gradients[d_str])
        axes[r, c].imshow(grad, cmap='viridis')
        axes[r, c].set_title(f"G_{d_str}° Direction", fontsize=12)
        axes[r, c].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def compare_sobel_methods(
    original: np.ndarray,
    two_dir_edges: np.ndarray,
    four_dir_edges: np.ndarray,
    eight_dir_edges: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Generate comparison plot matching Paper Figure 20:
    (a) Original Image
    (b) 2-Direction Sobel
    (c) 4-Direction Sobel
    (d) Proposed 8-Direction QSED
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("(a) Original Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(two_dir_edges, cmap='gray')
    axes[1].set_title("(b) 2-Direction QSED [Fan 2019]", fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(four_dir_edges, cmap='gray')
    axes[2].set_title("(c) 4-Direction QSED [Chetia 2021]", fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(eight_dir_edges, cmap='gray')
    axes[3].set_title("(d) Proposed 8-Direction QSED", fontsize=14)
    axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
