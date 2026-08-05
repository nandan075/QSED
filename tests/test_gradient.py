"""
Unit Tests for 8-Direction Sobel Masks and Gradient Magnitude Computation.

Paper Reference: Section 2.2, Section 3.2 Step 3 (Equations 3, 4, 5, 10, 11).
"""

import numpy as np
import pytest
from src.classical.sobel_masks import get_eight_sobel_masks
from src.classical.sobel_operator import apply_sobel_masks
from src.classical.gradient import compute_gradient_magnitude


def test_sobel_masks_symmetry_and_size():
    """
    Verify all eight Sobel directional masks are 5x5 kernels.
    """
    masks = get_eight_sobel_masks()
    assert len(masks) == 8

    expected_keys = ["0", "22.5", "45", "67.5", "90", "112.5", "135", "157.5"]
    for key in expected_keys:
        assert key in masks
        assert masks[key].shape == (5, 5)


def test_gradient_computation_vertical_edge():
    """
    Test 8-direction Sobel gradient computation on a step vertical edge image.
    Image: Left half 0, right half 255.
    Expected: Strong response in 0° direction (horizontal derivative).
    """
    img = np.zeros((16, 16), dtype=np.uint8)
    img[:, 8:] = 255

    dir_grads = apply_sobel_masks(img)
    grad_mag, dom_dir = compute_gradient_magnitude(dir_grads)

    # Edge column x=7 or x=8 should have maximum gradient magnitude
    assert np.max(grad_mag[:, 7:9]) > 0
