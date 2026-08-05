"""
Image Loader Module for Quantum Image Edge Detection (QSED).

This module handles image reading, resizing to power-of-two dimensions (2^n x 2^n),
grayscale normalization (q-bit precision), synthetic test image generation, and saving outputs.

Paper Reference:
- Section 2.1: NEQR Quantum Image Representation (Requires 2^n x 2^n images with q-bit grayscale).
- Section 4.2: Simulation Experiments (Uses standard test images of size 512x512).
"""

import os
from typing import Tuple, Optional
import numpy as np
from PIL import Image


def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    q_bits: int = 8
) -> np.ndarray:
    """
    Load an image from disk, convert it to grayscale, resize to target power-of-two dimensions,
    and normalize pixel values to range [0, 2^q - 1].

    Purpose:
        Prepare digital image matrices for NEQR encoding and classical comparison pipelines.
        NEQR requires spatial dimensions of 2^n x 2^n and grayscale values bounded by 2^q levels.

    Inputs:
        image_path (str): Absolute or relative filepath to input image.
        target_size (Optional[Tuple[int, int]]): Target (height, width). Must be power of 2.
            If None, image is kept at current size (or resized to nearest power of 2).
        q_bits (int): Bit-depth of grayscale intensity (default: 8 bits -> [0, 255]).

    Outputs:
        np.ndarray: Grayscale pixel matrix of shape (2^n, 2^n) with dtype np.uint8.

    Time Complexity:
        O(H * W) where H, W are image dimensions due to reading, resampling, and pixel normalization.

    Space Complexity:
        O(2^n * 2^n) storage for the output image matrix.

    Reference:
        Paper Section 2.1 (NEQR model requires 2^n x 2^n resolution and q-bit intensity range).
    """
    # Load image from filesystem using PIL
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    with Image.open(image_path) as img:
        # Convert RGB/RGBA to single-channel 8-bit L mode
        img_gray = img.convert('L')

        # Compute or enforce target size 2^n x 2^n
        if target_size is not None:
            h, w = target_size
            # Ensure dimensions are powers of 2
            if (h & (h - 1) != 0) or (w & (w - 1) != 0) or (h != w):
                raise ValueError(f"Target size must be square and power of 2, got {target_size}")
            img_gray = img_gray.resize((w, h), Image.Resampling.BILINEAR)
        else:
            w, h = img_gray.size
            # Resize to nearest power of 2 square if not already
            n_h = max(1, int(2 ** np.round(np.log2(h))))
            n_w = max(1, int(2 ** np.round(np.log2(w))))
            side = min(n_h, n_w)
            if (h != side) or (w != side):
                img_gray = img_gray.resize((side, side), Image.Resampling.BILINEAR)

        img_array = np.array(img_gray, dtype=np.float64)

        # Scale intensity to q-bit range [0, 2^q - 1]
        max_val = (1 << q_bits) - 1
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * max_val

        return np.clip(img_array, 0, max_val).astype(np.uint8)


def save_image(image_matrix: np.ndarray, output_path: str) -> None:
    """
    Save a numpy grayscale image matrix to a PNG file on disk.

    Purpose:
        Export output edge maps, gradient images, NMS images, and threshold results.

    Inputs:
        image_matrix (np.ndarray): 2D array of pixel values.
        output_path (str): Destination file path.

    Outputs:
        None (Writes image file to disk).

    Time Complexity:
        O(H * W) to encode PNG format and write to filesystem.

    Space Complexity:
        O(H * W) buffer for file writing.

    Reference:
        Paper Section 4.2 (Exporting experimental output images).
    """
    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Normalize image to uint8 for PNG writing
    if image_matrix.dtype != np.uint8:
        norm = image_matrix.astype(np.float64)
        if norm.max() > 0:
            norm = (norm / norm.max()) * 255.0
        img_uint8 = np.clip(norm, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image_matrix

    img = Image.fromarray(img_uint8)
    img.save(output_path)


def generate_synthetic_image(size: int = 16, pattern: str = "checkerboard") -> np.ndarray:
    """
    Generate synthetic test images (e.g. checkerboard, circle, diagonal line, square) for testing quantum pipelines.

    Purpose:
        Provide deterministic test images of controllable 2^n x 2^n sizes to test quantum circuit execution
        and verify edge detection accuracy against known ground truth edges.

    Inputs:
        size (int): Image side length (must be power of 2, e.g., 4, 8, 16, 32).
        pattern (str): 'checkerboard', 'circle', 'diagonal', or 'square'.

    Outputs:
        np.ndarray: Synthetic grayscale image matrix of shape (size, size) with uint8 values in [0, 255].

    Time Complexity:
        O(N^2) where N = size.

    Space Complexity:
        O(N^2) for the image matrix array.

    Reference:
        Paper Section 4.2 (Synthetic verification of 2^n x 2^n quantum images).
    """
    if (size & (size - 1) != 0) or size < 2:
        raise ValueError(f"Size must be a power of 2 and >= 2, got {size}")

    img = np.zeros((size, size), dtype=np.uint8)

    if pattern == "checkerboard":
        block_size = max(1, size // 4)
        for r in range(size):
            for c in range(size):
                if ((r // block_size) + (c // block_size)) % 2 == 1:
                    img[r, c] = 255
                else:
                    img[r, c] = 0
    elif pattern == "square":
        margin = max(1, size // 4)
        img[margin:size-margin, margin:size-margin] = 255
    elif pattern == "circle":
        cy, cx = size // 2, size // 2
        radius = size // 3
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        img[dist_from_center <= radius] = 255
    elif pattern == "diagonal":
        for i in range(size):
            img[i, i] = 255
            if i + 1 < size:
                img[i, i + 1] = 200
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return img
