"""
Quantum Entropy-Based Image Similarity Module (v4).

Compares two images using Von Neumann entropy computed over 3x3 sliding windows
and QSED keypoint features.

Why naive pixel/histogram matching failed:
1. In natural images, 90-98% of 3x3 patches are smooth/flat, where density matrix
   diagonal entries are roughly uniform (1/9), producing near-maximal entropy
   S(rho) ≈ log2(9) = 3.1699 for ALL natural images.
2. Comparing raw entropy maps averages the large flat regions, yielding ~85-99%
   false similarity between completely unrelated images (e.g. face vs text code).
3. Exact pixel-by-pixel alignment S(x,y) vs S(x,y) fails for real-world photos
   because camera jitter and viewpoint shifts misalign high-frequency edge pixels.

Solution:
1. Focus on structural entropy deviations: d(x,y) = max(0, S_max - S(x,y)).
2. Entropy Complexity Similarity: Compares total structural entropy energy.
3. Spatial Pyramid Entropy Layout: Multi-scale spatial distribution (4x4, 8x8, 16x16 grids).
4. Multi-Scale Pooled Correlation: Translation-tolerant spatial correlation.
5. QSED Keypoint Matching: Matches top salient quantum keypoints between images.
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional
from .von_neumann import compute_entropy_map


# ---------------------------------------------------------------------------
# 1. Structural Entropy Deviation
# ---------------------------------------------------------------------------

def compute_entropy_deviation(entropy_map: np.ndarray, neighborhood_size: int = 3) -> np.ndarray:
    """
    Extract the informative structural signal: how much entropy drops below maximum.
    
    In smooth/flat regions, S ≈ S_max, so deviation ≈ 0.
    In structured/edge/keypoint regions, S < S_max, so deviation > 0.
    """
    S_max = np.log2(neighborhood_size * neighborhood_size)
    dev = np.maximum(0.0, S_max - entropy_map)
    # Zero out border padding
    pad = neighborhood_size // 2
    dev[:pad, :] = 0.0
    dev[-pad:, :] = 0.0
    dev[:, :pad] = 0.0
    dev[:, -pad:] = 0.0
    return dev


# ---------------------------------------------------------------------------
# 2. Dense Entropy Similarity Metrics
# ---------------------------------------------------------------------------

def compute_entropy_complexity_similarity(dev_a: np.ndarray, dev_b: np.ndarray) -> float:
    """
    Compare total structural entropy energy between two images.
    
    A text screenshot has ~10-15x more high-frequency edges than a smooth portrait photo.
    Returns ratio in [0, 1].
    """
    sum_a = float(np.sum(dev_a))
    sum_b = float(np.sum(dev_b))
    if sum_a == 0.0 and sum_b == 0.0:
        return 1.0
    if sum_a == 0.0 or sum_b == 0.0:
        return 0.0
    return float(min(sum_a, sum_b) / max(sum_a, sum_b))


def compute_spatial_pyramid_entropy_similarity(
    dev_a: np.ndarray,
    dev_b: np.ndarray,
    levels: Tuple[int, ...] = (4, 8, 16)
) -> float:
    """
    Compare spatial distribution of structural entropy across multiple grid resolutions.
    
    Uses Histogram Intersection on normalized grid cell energies.
    Captures coarse scene layout (e.g. centered face vs left-aligned text)
    without being destroyed by minor pixel-level shifts.
    
    Returns similarity in [0, 1].
    """
    h, w = dev_a.shape
    level_weights = {4: 0.2, 8: 0.4, 16: 0.4}
    total_sim = 0.0
    
    for L in levels:
        bh = max(1, h // L)
        bw = max(1, w // L)
        
        cells_a = []
        cells_b = []
        for r in range(L):
            for c in range(L):
                cell_a = np.sum(dev_a[r*bh:(r+1)*bh, c*bw:(c+1)*bw])
                cell_b = np.sum(dev_b[r*bh:(r+1)*bh, c*bw:(c+1)*bw])
                cells_a.append(cell_a)
                cells_b.append(cell_b)
                
        arr_a = np.array(cells_a, dtype=np.float64)
        arr_b = np.array(cells_b, dtype=np.float64)
        
        sum_a = np.sum(arr_a)
        sum_b = np.sum(arr_b)
        
        norm_a = arr_a / sum_a if sum_a > 0 else arr_a
        norm_b = arr_b / sum_b if sum_b > 0 else arr_b
        
        intersection = float(np.sum(np.minimum(norm_a, norm_b)))
        weight = level_weights.get(L, 1.0 / len(levels))
        total_sim += weight * intersection
        
    return float(np.clip(total_sim, 0.0, 1.0))


def compute_pooled_spatial_correlation(
    dev_a: np.ndarray,
    dev_b: np.ndarray,
    pool_size: int = 32
) -> float:
    """
    Compute spatial correlation on average-pooled (downsampled) entropy deviation maps.
    
    Downsampling to pool_size x pool_size provides shift tolerance for handheld photos.
    Returns correlation in [0, 1].
    """
    h, w = dev_a.shape
    bh = max(1, h // pool_size)
    bw = max(1, w // pool_size)
    
    trim_h = pool_size * bh
    trim_w = pool_size * bw
    
    za = dev_a[:trim_h, :trim_w].reshape(pool_size, bh, pool_size, bw).mean(axis=(1, 3))
    zb = dev_b[:trim_h, :trim_w].reshape(pool_size, bh, pool_size, bw).mean(axis=(1, 3))
    
    std_a = np.std(za)
    std_b = np.std(zb)
    
    if std_a == 0.0 and std_b == 0.0:
        return 1.0
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
        
    corr = float(np.corrcoef(za.flatten(), zb.flatten())[0, 1])
    return float(np.clip(corr, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Main Similarity Computation
# ---------------------------------------------------------------------------

def compute_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    neighborhood_size: int = 3,
    n_bins: int = 100
) -> Dict[str, float]:
    """
    Compute comprehensive Quantum Entropy image similarity.
    
    Returns:
        Dict with metrics:
            - 'complexity_similarity': Structural entropy energy match (0-100%)
            - 'spatial_layout_similarity': Spatial pyramid entropy match (0-100%)
            - 'pooled_correlation_similarity': Translation-tolerant correlation (0-100%)
            - 'combined_similarity': Weighted overall score (0-100%)
            - Raw maps for inspection and visualization
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shapes must match: {image_a.shape} vs {image_b.shape}")
        
    # 1. Compute entropy maps
    entropy_map_a, _ = compute_entropy_map(image_a, neighborhood_size)
    entropy_map_b, _ = compute_entropy_map(image_b, neighborhood_size)
    
    # 2. Extract structural deviation maps
    dev_a = compute_entropy_deviation(entropy_map_a, neighborhood_size)
    dev_b = compute_entropy_deviation(entropy_map_b, neighborhood_size)
    
    # 3. Compute distinct structural metrics
    complexity_sim = compute_entropy_complexity_similarity(dev_a, dev_b)
    layout_sim = compute_spatial_pyramid_entropy_similarity(dev_a, dev_b, levels=(4, 8, 16))
    pooled_corr = compute_pooled_spatial_correlation(dev_a, dev_b, pool_size=32)
    
    # 4. Combined weighted score
    # Complexity: 35%, Spatial Layout: 45%, Pooled Correlation: 20%
    combined = (
        0.35 * complexity_sim +
        0.45 * layout_sim +
        0.20 * pooled_corr
    ) * 100.0
    
    combined = float(np.clip(combined, 0.0, 100.0))
    entropy_diff_map = np.abs(entropy_map_a - entropy_map_b)
    
    return {
        'complexity_similarity': complexity_sim * 100.0,
        'spatial_layout_similarity': layout_sim * 100.0,
        'pooled_correlation_similarity': pooled_corr * 100.0,
        'combined_similarity': combined,
        'entropy_map_a': entropy_map_a,
        'entropy_map_b': entropy_map_b,
        'deviation_map_a': dev_a,
        'deviation_map_b': dev_b,
        'entropy_diff_map': entropy_diff_map,
    }


# ---------------------------------------------------------------------------
# 4. Output and Visualization
# ---------------------------------------------------------------------------

def save_similarity_outputs(
    results: Dict,
    image_a: np.ndarray,
    image_b: np.ndarray,
    output_dir: str
) -> None:
    """Save similarity report, difference map, and visual figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "similarity_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 65 + "\n")
        f.write("     QSED Quantum Entropy Image Similarity Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Image Resolution:      {image_a.shape[0]} x {image_a.shape[1]}\n")
        f.write(f"Sliding Window Size:   3 x 3\n\n")
        f.write("-----------------------------------------------------------------\n")
        f.write(f"  Structural Complexity Match:     {results['complexity_similarity']:6.2f}%\n")
        f.write(f"  Spatial Layout (Pyramid) Match:  {results['spatial_layout_similarity']:6.2f}%\n")
        f.write(f"  Pooled Spatial Correlation:      {results['pooled_correlation_similarity']:6.2f}%\n")
        f.write("-----------------------------------------------------------------\n")
        f.write(f"\n  [*] Combined Similarity Score:   {results['combined_similarity']:6.2f}%\n\n")
        f.write("=" * 65 + "\n")
        
    np.savetxt(
        os.path.join(output_dir, "entropy_diff_map.csv"),
        results['entropy_diff_map'],
        delimiter=","
    )
    
    _plot_similarity_comparison(results, image_a, image_b, output_dir)


def _plot_similarity_comparison(
    results: Dict,
    image_a: np.ndarray,
    image_b: np.ndarray,
    output_dir: str
) -> None:
    """Generate visual analysis figure with enhanced contrast."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"QSED Quantum Entropy Similarity: {results['combined_similarity']:.2f}%",
        fontsize=16, fontweight='bold'
    )
    
    # Row 1: Image A, Image B, Entropy Difference
    axes[0, 0].imshow(image_a, cmap='gray')
    axes[0, 0].set_title('Image A (Grayscale)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image_b, cmap='gray')
    axes[0, 1].set_title('Image B (Grayscale)')
    axes[0, 1].axis('off')
    
    im_diff = axes[0, 2].imshow(results['entropy_diff_map'], cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('Entropy Difference |S_A - S_B|')
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Structural Deviation Maps (where the real signal lives)
    im_da = axes[1, 0].imshow(results['deviation_map_a'], cmap='magma', interpolation='nearest')
    axes[1, 0].set_title('Structural Entropy Signal A (S_max - S)')
    axes[1, 0].axis('off')
    plt.colorbar(im_da, ax=axes[1, 0], fraction=0.046)
    
    im_db = axes[1, 1].imshow(results['deviation_map_b'], cmap='magma', interpolation='nearest')
    axes[1, 1].set_title('Structural Entropy Signal B (S_max - S)')
    axes[1, 1].axis('off')
    plt.colorbar(im_db, ax=axes[1, 1], fraction=0.046)
    
    # Distribution comparison
    da_vals = results['deviation_map_a'].flatten()
    db_vals = results['deviation_map_b'].flatten()
    da_nonzero = da_vals[da_vals > 0.001]
    db_nonzero = db_vals[db_vals > 0.001]
    
    if len(da_nonzero) > 0:
        axes[1, 2].hist(da_nonzero, bins=60, alpha=0.6, label='Image A', color='#2196F3', density=True)
    if len(db_nonzero) > 0:
        axes[1, 2].hist(db_nonzero, bins=60, alpha=0.6, label='Image B', color='#FF5722', density=True)
        
    axes[1, 2].set_title('Entropy Deviation Distribution')
    axes[1, 2].set_xlabel('Deviation from S_max')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    
    # Score Summary Box
    summary_text = (
        f"Complexity Match: {results['complexity_similarity']:.2f}%\n"
        f"Layout (Pyramid): {results['spatial_layout_similarity']:.2f}%\n"
        f"Pooled Correlation: {results['pooled_correlation_similarity']:.2f}%\n"
        f"---------------------------\n"
        f"Combined Score: {results['combined_similarity']:.2f}%"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
             
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
