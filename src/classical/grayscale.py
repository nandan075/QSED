"""
Grayscale Conversion Module for Quantum Image Edge Detection (QSED).

This module provides standard luminance-based grayscale conversion and bit-depth quantization.

Paper Reference:
- Section 2.1: NEQR model uses q-bit grayscale intensity values C_YX in range [0, 2^q - 1].
"""

from typing import Union
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB or RGBA image matrix into a 2D grayscale intensity matrix using standard ITU-R BT.601 luminance weights.

    Purpose:
        Single-channel conversion of color image input. NEQR models 2D scalar pixel intensity fields.

    Inputs:
        image (np.ndarray): Input image array of shape (H, W), (H, W, 3) or (H, W, 4).

    Outputs:
        np.ndarray: Grayscale matrix of shape (H, W) with dtype np.uint8.

    Time Complexity:
        O(H * W) linear iteration over image pixels.

    Space Complexity:
        O(H * W) allocation for grayscale result matrix.

    Reference:
        Paper Section 2.1 (NEQR scalar grayscale encoding).
    """
    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.ndim == 3:
        if image.shape[2] == 3:
            # Luminance formula: Y = 0.299 R + 0.587 G + 0.114 B
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return np.clip(gray, 0, 255).astype(np.uint8)
        elif image.shape[2] == 4:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            return np.clip(gray, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def quantize_grayscale(image: np.ndarray, q_bits: int = 8) -> np.ndarray:
    """
    Quantize pixel intensity values to q bits precision [0, 2^q - 1].

    Purpose:
        Match the q-bit grayscale encoding register of NEQR representation: |C_YX> = |C_{q-1} ... C_0>.

    Inputs:
        image (np.ndarray): Grayscale image matrix.
        q_bits (int): Number of bits q per pixel intensity (typically 8 for 256 gray levels).

    Outputs:
        np.ndarray: Quantized grayscale image of dtype np.uint8.

    Time Complexity:
        O(H * W) element-wise transformation.

    Space Complexity:
        O(H * W) space for output matrix.

    Reference:
        Paper Section 2.1, Equation (1).
    """
    max_val = (1 << q_bits) - 1
    norm = image.astype(np.float64)
    if norm.max() > 0:
        norm = (norm / norm.max()) * max_val
    return np.clip(np.round(norm), 0, max_val).astype(np.uint8)
