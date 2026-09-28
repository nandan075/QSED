"""
Double Threshold Detection Module for Quantum Image Edge Detection (QSED).

Categorizes NMS-suppressed pixels into strong edge points, weak edge points, and non-edge points
using high (T_H) and low (T_L) threshold values.

Paper Reference:
- Section 3.2: Step 5 Double threshold detection (Equation 14, Figure 18).
"""

from typing import Tuple, Optional
import numpy as np


def double_threshold(
    nms_image: np.ndarray,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    high_ratio: float = 0.15,
    low_ratio: float = 0.05
) -> Tuple[np.ndarray, float, float]:
    """
    Apply double thresholding to categorize pixels into strong, weak, and non-edge points.

    Purpose:
        Filter out noise and false edges using dual intensity thresholds T_H and T_L.
        - Strong edge (E_YX = 10, encoded as 2): G >= T_H. High confidence real edges.
        - Weak edge (E_YX = 01, encoded as 1): T_L <= G < T_H. Candidate edges to be tracked.
        - Non-edge (E_YX = 00, encoded as 0): G < T_L. Suppressed background/noise.

    Inputs:
        nms_image (np.ndarray): NMS-processed gradient image.
        high_threshold (Optional[float]): Explicit T_H value. If None, computed automatically from max gradient.
        low_threshold (Optional[float]): Explicit T_L value. If None, set to T_H / 3 per Paper Step 5.
        high_ratio (float): Fraction of max gradient to set T_H if high_threshold is None.
        low_ratio (float): Ratio for low threshold (paper specifies T_L = T_H / 3).

    Outputs:
        Tuple[np.ndarray, float, float]:
            - edge_map (np.ndarray): 2D array of shape (H, W) where values are:
                2 for strong edges (binary '10'),
                1 for weak edges (binary '01'),
                0 for non-edges (binary '00').
            - high_threshold (float): Effective T_H value used.
            - low_threshold (float): Effective T_L value used.

    Time Complexity:
        O(H * W) element-wise conditional evaluation.

    Space Complexity:
        O(H * W) space for edge state map.

    Reference:
        Paper Section 3.2, Step 5, Equation (14), Figure 18.
        Note: The paper explicitly states T_L = (1/3) * T_H.
    """
    max_val = np.max(nms_image)

    if high_threshold is None:
        high_threshold = max_val * high_ratio

    if low_threshold is None:
        # Paper explicitly sets T_L = (1/3) * T_H
        low_threshold = high_threshold / 3.0

    h, w = nms_image.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)

    # Strong edge points: gradient >= T_H -> E_YX = 10 (decimal 2)
    strong_mask = (nms_image >= high_threshold)
    edge_map[strong_mask] = 2

    # Weak edge points: T_L <= gradient < T_H -> E_YX = 01 (decimal 1)
    weak_mask = (nms_image >= low_threshold) & (nms_image < high_threshold)
    edge_map[weak_mask] = 1

    # Non-edge points: gradient < T_L -> E_YX = 00 (decimal 0)
    # Remaining locations are already 0

    return edge_map, high_threshold, low_threshold
