import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from .keypoint_detection import Keypoint

def plot_keypoint_analysis(
    original: np.ndarray,
    edge_map: np.ndarray,
    dominant_direction: np.ndarray,
    directional_variation_map: np.ndarray,
    keypoint_score_map: np.ndarray,
    entropy_map: np.ndarray,
    keypoints: List[Keypoint],
    save_dir: str = None
) -> None:
    """
    Plot a 3x3 keypoint analysis figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # 1. Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')
    
    # 2. Final edge map
    axes[1].imshow(edge_map, cmap='gray')
    axes[1].set_title("2. Final Edge Map")
    axes[1].axis('off')
    
    # 3. Dominant-direction map
    axes[2].imshow(dominant_direction, cmap='hsv', vmin=0, vmax=7)
    axes[2].set_title("3. Dominant Direction")
    axes[2].axis('off')
    
    # 4. Directional variation map
    axes[3].imshow(directional_variation_map, cmap='hot')
    axes[3].set_title("4. Directional Variation Map")
    axes[3].axis('off')
    
    # 5. Keypoint score map
    axes[4].imshow(keypoint_score_map, cmap='inferno')
    axes[4].set_title("5. Keypoint Score Map")
    axes[4].axis('off')
    
    # 6. Keypoints overlaid on original
    axes[5].imshow(original, cmap='gray')
    axes[5].set_title("6. Keypoints on Original")
    for kp in keypoints:
        axes[5].plot(kp.x, kp.y, 'ro', markersize=4)
    axes[5].axis('off')
    
    # 7. Keypoints overlaid on edge map
    axes[6].imshow(edge_map, cmap='gray')
    axes[6].set_title("7. Keypoints on Edges")
    for kp in keypoints:
        axes[6].plot(kp.x, kp.y, 'co', markersize=4)
    axes[6].axis('off')
    
    # 8. Full-image entropy heatmap
    axes[7].imshow(entropy_map, cmap='viridis')
    axes[7].set_title("8. Full-image Entropy Heatmap")
    axes[7].axis('off')
    
    # 9. Top-ranked keypoints (top 30) with entropy value annotations overlaid on entropy heatmap
    axes[8].imshow(entropy_map, cmap='viridis')
    axes[8].set_title("9. Top 30 Keypoints on Entropy Heatmap")
    top_30 = sorted(keypoints, key=lambda k: k.combined_score, reverse=True)[:30]
    for kp in top_30:
        axes[8].plot(kp.x, kp.y, 'r*', markersize=6)
        axes[8].text(kp.x + 1, kp.y, f'{kp.entropy:.2f}', color='white', fontsize=8)
    axes[8].axis('off')
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'keypoint_analysis.png'), dpi=300)
    plt.close()

def plot_entropy_comparison(
    entropy_map_a: np.ndarray,
    entropy_map_b: np.ndarray,
    keypoints: List[Keypoint],
    save_path: str = None
) -> None:
    """
    Plot side-by-side comparison of Method A vs Method B entropy maps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(entropy_map_a, cmap='viridis')
    ax1.set_title("Method A Entropy Map")
    fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(entropy_map_b, cmap='viridis')
    ax2.set_title("Method B Entropy Map")
    fig.colorbar(im2, ax=ax2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
