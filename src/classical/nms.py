"""
Non-Maximum Suppression (NMS) Module for Quantum Image Edge Detection (QSED).

Suppresses non-peak gradient responses by comparing each pixel's gradient magnitude
against its two neighbors along the dominant gradient direction within the 5x5 window.

Paper Reference:
- Section 3.2: Step 4 Non-maximum suppression (Equation 13, Figure 17).
"""

from typing import Tuple
import numpy as np


def non_maximum_suppression(
    gradient_magnitude: np.ndarray,
    dominant_direction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Non-Maximum Suppression (NMS) on gradient magnitude image using 8 gradient directions.

    Purpose:
        Thin out thick edges and remove false edge detections. A pixel is retained if its gradient magnitude
        is greater than or equal to both neighbor pixels along its dominant gradient orientation.

    Inputs:
        gradient_magnitude (np.ndarray): 2D matrix of maximum gradient values G(Y,X).
        dominant_direction (np.ndarray): 2D matrix of direction indices (0..7).

    Outputs:
        Tuple[np.ndarray, np.ndarray]:
            - nms_image (np.ndarray): Suppressed gradient image of shape (H, W).
            - max_mask (np.ndarray): Binary mask |M> where 1 indicates local maximum pixel, 0 non-maximum.

    Time Complexity:
        O(H * W) iteration over pixels.

    Space Complexity:
        O(H * W) for NMS result image and binary mask.

    Reference:
        Paper Section 3.2, Step 4, Equation (13), Figure 17.
    """
    h, w = gradient_magnitude.shape
    nms_image = np.zeros((h, w), dtype=np.float64)
    max_mask = np.zeros((h, w), dtype=np.uint8)

    # Offset lookup for neighbors along the 8 orientations within 5x5 window
    # Index 0: 0° (horizontal) -> neighbors (0, -1) and (0, 1)
    # Index 1: 22.5° -> neighbors (-1, -2) and (1, 2)
    # Index 2: 45° (diagonal) -> neighbors (-1, -1) and (1, 1)
    # Index 3: 67.5° -> neighbors (-2, -1) and (2, 1)
    # Index 4: 90° (vertical) -> neighbors (-1, 0) and (1, 0)
    # Index 5: 112.5° -> neighbors (-2, 1) and (2, -1)
    # Index 6: 135° (anti-diagonal) -> neighbors (-1, 1) and (1, -1)
    # Index 7: 157.5° -> neighbors (-1, 2) and (1, -2)
    offsets = [
        ((0, -1), (0, 1)),       # 0°
        ((-1, -2), (1, 2)),     # 22.5°
        ((-1, -1), (1, 1)),     # 45°
        ((-2, -1), (2, 1)),     # 67.5°
        ((-1, 0), (1, 0)),       # 90°
        ((-2, 1), (2, -1)),     # 112.5°
        ((-1, 1), (1, -1)),     # 135°
        ((-1, 2), (1, -2))      # 157.5°
    ]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            direction = dominant_direction[y, x]
            off1, off2 = offsets[direction]

            val = gradient_magnitude[y, x]
            val1 = gradient_magnitude[y + off1[0], x + off1[1]]
            val2 = gradient_magnitude[y + off2[0], x + off2[1]]

            # Retain pixel if it is greater than or equal to both neighbors
            if val >= val1 and val >= val2 and val > 0:
                nms_image[y, x] = val
                max_mask[y, x] = 1
            else:
                nms_image[y, x] = 0
                max_mask[y, x] = 0

    return nms_image, max_mask
