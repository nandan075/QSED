"""
Edge Tracking Module for Quantum Image Edge Detection (QSED).

Performs hysteresis edge tracking: converts weak edge points into true edges if at least one
strong edge point exists within their 24-neighborhood (5x5 window).

Paper Reference:
- Section 3.2: Step 6 Edge tracking (Equation 15, Figure 19).
"""

import numpy as np


def edge_tracking_hysteresis(threshold_map: np.ndarray) -> np.ndarray:
    """
    Perform 24-neighborhood hysteresis edge tracking on double-thresholded image.

    Purpose:
        Determine whether weak candidate edge pixels (E_YX = 01) are connected to true strong edges (E_YX = 10).
        Per Section 3.2 Step 6, the paper checks a 24-neighborhood (5x5 pixel window excluding center) for any
        strong edge point. If present, the weak edge is confirmed as a true edge (|B_YX> = |1>).
        Otherwise, it is discarded as noise (|B_YX> = |0>).

    Inputs:
        threshold_map (np.ndarray): 2D array of shape (H, W) with values:
            2 = Strong edge (E_YX = 10),
            1 = Weak edge (E_YX = 01),
            0 = Non-edge (E_YX = 00).

    Outputs:
        np.ndarray: Binary final edge map of shape (H, W) where 1 indicates final edge pixel (|B_YX> = |1>)
        and 0 indicates background pixel (|B_YX> = |0>).

    Time Complexity:
        O(H * W) iteration with BFS / neighborhood search.

    Space Complexity:
        O(H * W) output binary edge matrix.

    Reference:
        Paper Section 3.2, Step 6, Equation (15), Figure 19.
    """
    h, w = threshold_map.shape
    final_edges = np.zeros((h, w), dtype=np.uint8)

    # Initialize all strong edge points as confirmed true edges
    strong_y, strong_x = np.where(threshold_map == 2)
    for y, x in zip(strong_y, strong_x):
        final_edges[y, x] = 1

    # 24-neighborhood window offsets (-2 to +2 in y and x, excluding center (0,0))
    window_offsets = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if not (dy == 0 and dx == 0)
    ]

    # Iterative hysteresis propagation until convergence
    changed = True
    while changed:
        changed = False
        weak_y, weak_x = np.where((threshold_map == 1) & (final_edges == 0))
        for y, x in zip(weak_y, weak_x):
            # Check 24-neighborhood for any confirmed strong edge
            has_strong_neighbor = False
            for dy, dx in window_offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if final_edges[ny, nx] == 1:
                        has_strong_neighbor = True
                        break

            if has_strong_neighbor:
                final_edges[y, x] = 1
                changed = True

    return final_edges
