"""
Image Loading and Preprocessing Module.

Handles:
- Loading images from local relative filesystem paths via PIL or OpenCV.
- Color conversion to single-channel grayscale float64 [0.0, 1.0] or uint8 [0, 255].
- Robust reflection padding with NumPy's `mode='reflect'` preserving boundary continuity.
- Synthetic image generator for deterministic offline testing.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image


def load_grayscale_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Load an image from disk and convert to grayscale.

    Args:
        image_path: Relative or absolute path to image file.
        target_size: Optional (height, width) tuple to resize image.
        normalize: If True, scales pixel intensities to [0.0, 1.0] float64.
                   If False, returns uint8 [0, 255].

    Returns:
        2D numpy array of shape (H, W).

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file is not a readable 2D image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            if target_size is not None:
                h, w = target_size
                gray = gray.resize((w, h), Image.Resampling.BILINEAR)
            arr = np.array(gray, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Failed to read image at '{image_path}': {e}") from e

    if normalize:
        max_val = np.max(arr)
        if max_val > 0.0:
            arr = arr / max_val
        return arr.astype(np.float64)

    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_boundary_padding(
    image: np.ndarray,
    pad_width: int,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply reflection padding to image boundaries.

    In 'reflect' mode (NumPy default), values mirror without duplicating the edge:
        [P1, P2, P3, P4] -> [P3, P2 | P1, P2, P3, P4 | P3, P2]

    Args:
        image: 2D array of shape (H, W).
        pad_width: Number of padding pixels along each border.
        mode: NumPy padding mode (default: 'reflect').

    Returns:
        Padded array of shape (H + 2*pad_width, W + 2*pad_width).
    """
    if pad_width <= 0:
        return image.copy()
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got array with shape {image.shape}")
    return np.pad(image, pad_width=pad_width, mode=mode)


def extract_square_patch(
    padded_image: np.ndarray,
    center_x: int,
    center_y: int,
    half_size: int,
) -> np.ndarray:
    """
    Extract a (2*half_size + 1) x (2*half_size + 1) patch from padded image.

    Args:
        padded_image: 2D array that already has reflection padding applied.
        center_x: Column coordinate in padded coordinates.
        center_y: Row coordinate in padded coordinates.
        half_size: Radius of the patch (e.g. 4 for 9x9 patch).

    Returns:
        Square patch of shape (2*half_size + 1, 2*half_size + 1).
    """
    r_min = center_y - half_size
    r_max = center_y + half_size + 1
    c_min = center_x - half_size
    c_max = center_x + half_size + 1
    return padded_image[r_min:r_max, c_min:c_max].copy()


def generate_synthetic_test_image(
    size: Tuple[int, int] = (64, 64),
    pattern: str = "checkerboard",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate synthetic deterministic test images for test suites and offline verification.

    Patterns:
    - 'checkerboard': High-frequency alternating block corners.
    - 'square': Centered white geometric square on black background.
    - 'circle': Centered smooth disc.
    - 'ramp': Linear gradient ramp.
    - 'random': Gaussian noise pattern with smooth texture.

    Returns:
        2D uint8 numpy array of shape (H, W).
    """
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)

    if pattern == "checkerboard":
        block = max(2, min(h, w) // 8)
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = ((y_grid // block) + (x_grid // block)) % 2 == 1
        img[mask] = 255.0

    elif pattern == "square":
        margin_y = h // 4
        margin_x = w // 4
        img[margin_y : h - margin_y, margin_x : w - margin_x] = 255.0

    elif pattern == "circle":
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 3
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        img[dist <= radius] = 255.0

    elif pattern == "ramp":
        ramp = np.linspace(0, 255, w)
        img = np.tile(ramp, (h, 1))

    elif pattern == "random":
        rng = np.random.RandomState(seed)
        img = rng.uniform(0, 255, size=(h, w))

    else:
        raise ValueError(f"Unknown synthetic pattern: {pattern}")

    return np.clip(img, 0, 255).astype(np.uint8)
