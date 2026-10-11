"""
Visualization Module for Pairwise Von Neumann Entropy Keypoint Matching.

Generates:
1. Side-by-side keypoint match visualization with green inlier lines and red outlier lines.
2. Pairwise Von Neumann Entropy Matrix Heatmaps E^A and E^B showing symmetric relational geometry.
3. Single Keypoint Relational Entropy Profile (polar or 1D bar chart).
4. Composite matching dashboard figure.
"""

import os
from typing import Optional, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .matcher import MatchResult
from .descriptor_builder import ImageEntropyRepresentation


def plot_pairwise_entropy_matrices(
    rep_a: ImageEntropyRepresentation,
    rep_b: ImageEntropyRepresentation,
    save_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot heatmaps of the symmetric pairwise entropy matrices E^A and E^B.

    Args:
        rep_a: Image A entropy representation.
        rep_b: Image B entropy representation.
        save_path: Optional path to save figure.
        title_suffix: Additional title text.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Matrix A
    mat_a = rep_a.entropy_matrix
    n_a = rep_a.num_keypoints
    im0 = axes[0].imshow(mat_a, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Image A: Pairwise Entropy Matrix E^A ({n_a} x {n_a})\nDiagonal = 0.0 (storage convention)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Keypoint Index j", fontsize=10)
    axes[0].set_ylabel("Keypoint Index i", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Entropy S_{ij} [bits]")

    # Matrix B
    mat_b = rep_b.entropy_matrix
    n_b = rep_b.num_keypoints
    im1 = axes[1].imshow(mat_b, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Image B: Pairwise Entropy Matrix E^B ({n_b} x {n_b})\nDiagonal = 0.0 (storage convention)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Keypoint Index j", fontsize=10)
    axes[1].set_ylabel("Keypoint Index i", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Entropy S_{ij} [bits]")

    plt.suptitle(f"Pairwise Von Neumann Entropy Matrices {title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_match_correspondences(
    img_a: np.ndarray,
    img_b: np.ndarray,
    match_result: MatchResult,
    max_display: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render side-by-side keypoint correspondences.
    Inliers are plotted in solid green; outliers (rejected by RANSAC) are in dashed red.

    Args:
        img_a: Grayscale array of Image A.
        img_b: Grayscale array of Image B.
        match_result: Result object from matcher.
        max_display: Maximum number of match lines to render.
        save_path: Destination path for PNG figure.

    Returns:
        matplotlib Figure object.
    """
    ha, wa = img_a.shape
    hb, wb = img_b.shape
    h_max = max(ha, hb)

    composite = np.zeros((h_max, wa + wb), dtype=np.uint8)
    composite[:ha, :wa] = img_a
    composite[:hb, wa : wa + wb] = img_b

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(composite, cmap="gray")
    title = (
        f"Pairwise Von Neumann Entropy Keypoint Matches\n"
        f"Matches: {match_result.num_matches} | Inliers: {match_result.inlier_count} | "
        f"Mean Relational Error: {match_result.mean_entropy_error:.4f} bits | "
        f"Similarity: {match_result.similarity_score:.2f}% ({match_result.geometric_status})"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    pts_a = match_result.pts_a
    pts_b = match_result.pts_b
    inliers = match_result.inlier_mask
    n_matches = len(pts_a)

    display_count = min(max_display, n_matches)
    for idx in range(display_count):
        xa, ya = pts_a[idx]
        xb, yb = pts_b[idx]
        is_inlier = bool(inliers[idx]) if len(inliers) > idx else False

        xb_offset = xb + wa
        color = "#00E676" if is_inlier else "#FF1744"
        style = "-" if is_inlier else ":"
        alpha = 0.85 if is_inlier else 0.4
        width = 1.8 if is_inlier else 1.0

        ax.plot([xa, xb_offset], [ya, yb], style, color=color, linewidth=width, alpha=alpha)
        ax.plot(xa, ya, "o", color="#00E5FF", markersize=4)
        ax.plot(xb_offset, yb, "o", color="#FFD600", markersize=4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_single_keypoint_relational_profile(
    rep: ImageEntropyRepresentation,
    keypoint_index: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the (n - 1)-dimensional descriptor D(k_i) of a selected keypoint.

    Args:
        rep: Image entropy representation.
        keypoint_index: Index of keypoint to visualize.
        save_path: Optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    if rep.num_keypoints <= 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient keypoints for relational profile", ha="center", va="center")
        return fig

    idx = max(0, min(keypoint_index, rep.num_keypoints - 1))
    desc_obj = rep.descriptors[idx]
    values = desc_obj.descriptor
    n_rel = len(values)

    fig, ax = plt.subplots(figsize=(10, 4))
    x_axis = np.arange(n_rel)
    ax.bar(x_axis, values, color="#304FFE", edgecolor="black", alpha=0.85)
    ax.set_ylim([0.0, 1.05])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Equal superposition (0.5 bits)")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.0, label="Orthogonal pure states (1.0 bit)")
    ax.set_title(
        f"Keypoint k_{idx} (x={desc_obj.x:.1f}, y={desc_obj.y:.1f}) (n - 1)={n_rel}-D Descriptor Profile",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Paired Keypoint j (j != i)", fontsize=10)
    ax.set_ylabel("Pairwise Entropy S_{ij} [bits]", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
