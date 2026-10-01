"""
Unit Tests for 24-Dimensional Local Von Neumann Entropy Feature Descriptor.

Verifies:
- Test 1: 5x5 neighborhood (24 surrounding pixels, KP excluded, deterministic ordering)
- Test 2: P7 local 3x3 neighborhood (P1 P2 P3 / P6 P7 P8 / P11 P12 KP)
- Test 3: Boundary keypoint reflection padding
- Test 4: Descriptor size (shape == (24,))
- Test 5: Density matrix properties (Tr(rho) ≈ 1, Hermitian, PSD)
- Test 6: Entropy properties (finite, non-negative, bounded by log2(9))
- Test 7: Image similarity & matching pipeline
"""

import os
import sys
import pytest
import numpy as np

# Ensure QSED/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature.entropy_descriptor import (
    extract_entropy_descriptor,
    extract_all_entropy_descriptors,
    match_entropy_descriptors,
    compute_pairwise_distances,
    compute_image_similarity_entropy_descriptor,
    apply_reflection_padding,
    get_p_neighborhood_labels,
    extract_single_p_entropy,
    P_LABELS_RASTER,
    P_LABELS_CLOCKWISE,
    P_OFFSETS_DICT,
    P_OFFSETS_RASTER,
    OFFSET_TO_LABEL
)
from src.feature.density_matrix import verify_density_matrix
from src.feature.keypoint_detection import Keypoint


# =============================================================================
# Test 1 — 5x5 Neighborhood & Ordering
# =============================================================================

def test_1_neighborhood_5x5():
    """
    Test 1:
    Verify that:
    - Exactly 24 surrounding pixels are returned.
    - KP (0, 0) is excluded from P1...P24.
    - Ordering is deterministic and consistent.
    """
    # 1. Exactly 24 surrounding positions
    assert len(P_LABELS_RASTER) == 24, "Must have exactly 24 labels in raster order"
    assert len(P_OFFSETS_RASTER) == 24, "Must have exactly 24 offsets in raster order"
    assert len(P_OFFSETS_DICT) == 24, "Must have exactly 24 entries in offset dict"

    # 2. KP (0, 0) is excluded from all Pi
    for p_name, offset in P_OFFSETS_DICT.items():
        assert offset != (0, 0), f"{p_name} has offset (0, 0), which is reserved for KP!"

    assert (0, 0) not in P_OFFSETS_RASTER, "KP offset (0, 0) must not be in P_OFFSETS_RASTER"

    # 3. Offsets span the 5x5 grid [-2, 2] x [-2, 2] excluding center
    all_5x5_offsets = {(r - 2, c - 2) for r in range(5) for c in range(5)}
    all_5x5_offsets.remove((0, 0))  # Remove KP

    assert set(P_OFFSETS_RASTER) == all_5x5_offsets, "Offsets must cover the complete 5x5 grid excluding KP"

    # 4. Deterministic ordering check
    labels_run_1 = list(P_LABELS_RASTER)
    labels_run_2 = list(P_LABELS_RASTER)
    assert labels_run_1 == labels_run_2, "Ordering must be strictly deterministic"
    assert P_LABELS_RASTER[0] == "P1"
    assert P_LABELS_RASTER[23] == "P24"


# =============================================================================
# Test 2 — P7 Local 3x3 Neighborhood
# =============================================================================

def test_2_p7_neighborhood():
    """
    Test 2:
    Explicitly verify that the local 3x3 neighborhood for P7 contains:
        P1   P2   P3
        P6   P7   P8
        P11  P12  KP
    """
    p7_labels = get_p_neighborhood_labels("P7")

    expected = [
        ["P1",  "P2",  "P3"],
        ["P6",  "P7",  "P8"],
        ["P11", "P12", "KP"]
    ]

    assert p7_labels == expected, (
        f"P7 3x3 neighborhood mismatch!\nExpected: {expected}\nGot: {p7_labels}"
    )

    # Also verify the underlying coordinate offsets for each cell:
    dy_p7, dx_p7 = P_OFFSETS_DICT["P7"]
    assert (dy_p7, dx_p7) == (-1, -1), f"P7 must be at (-1, -1), got ({dy_p7}, {dx_p7})"

    for r_idx, d_row in enumerate((-1, 0, 1)):
        for c_idx, d_col in enumerate((-1, 0, 1)):
            offset = (dy_p7 + d_row, dx_p7 + d_col)
            cell_label = OFFSET_TO_LABEL[offset]
            assert cell_label == expected[r_idx][c_idx], (
                f"Offset {offset} gave {cell_label}, expected {expected[r_idx][c_idx]}"
            )


# =============================================================================
# Test 3 — Boundary Keypoint Reflection Padding
# =============================================================================

def test_3_boundary_keypoint():
    """
    Test 3:
    Place a keypoint close to an image boundary (including the exact corner (0, 0))
    and verify that reflection padding allows a complete 3x3 neighborhood for all 24 Pi.
    Also verify NumPy reflection padding behavior: [P1 P2 P3 P4] -> [P3 P2 | P1 P2 P3 P4 | P3 P2].
    """
    # 1. Verify 1D reflection padding non-duplication property
    arr_1d = np.array([[10, 20, 30, 40]], dtype=np.float64)
    padded_1d = np.pad(arr_1d, pad_width=((0, 0), (2, 2)), mode="reflect")
    expected_row = np.array([30, 20, 10, 20, 30, 40, 30, 20], dtype=np.float64)
    assert np.array_equal(padded_1d[0], expected_row), (
        f"Reflection padding must not duplicate boundary! Got {padded_1d[0]}, expected {expected_row}"
    )

    # 2. Keypoints on boundaries: (0, 0), (0, 15), (15, 0), (15, 15)
    img = np.arange(256, dtype=np.float64).reshape((16, 16))
    boundary_points = [(0, 0), (0, 15), (15, 0), (15, 15), (0, 7), (15, 7), (7, 0), (7, 15)]

    for x, y in boundary_points:
        desc = extract_entropy_descriptor(img, (x, y))
        assert desc.shape == (24,), f"Boundary keypoint at ({x}, {y}) failed shape check"
        assert not np.isnan(desc).any(), f"Boundary keypoint at ({x}, {y}) produced NaN"
        assert not np.isinf(desc).any(), f"Boundary keypoint at ({x}, {y}) produced Inf"
        assert np.all(desc >= 0.0), f"Boundary keypoint at ({x}, {y}) produced negative entropy"


# =============================================================================
# Test 4 — Descriptor Size
# =============================================================================

def test_4_descriptor_size():
    """
    Test 4:
    For every valid keypoint, assert descriptor.shape == (24,).
    For N keypoints, assert feature_matrix.shape == (N, 24).
    """
    img = np.random.RandomState(42).randint(0, 256, size=(32, 32)).astype(np.float64)

    # Test single keypoint
    kp = Keypoint(x=10, y=12, dominant_direction=45.0, gradient_magnitude=100.0,
                  directional_variation=0.5, keypoint_score=0.8)
    desc = extract_entropy_descriptor(img, kp)
    assert isinstance(desc, np.ndarray), "Descriptor must be a numpy ndarray"
    assert desc.shape == (24,), f"Single descriptor shape must be (24,), got {desc.shape}"

    # Test N keypoints
    kps = [Keypoint(x=i * 5 + 3, y=i * 4 + 3, dominant_direction=0.0, gradient_magnitude=1.0,
                    directional_variation=0.1, keypoint_score=0.5) for i in range(5)]
    mat = extract_all_entropy_descriptors(img, kps)
    assert mat.shape == (5, 24), f"Feature matrix shape must be (5, 24), got {mat.shape}"


# =============================================================================
# Test 5 — Density Matrix Properties
# =============================================================================

def test_5_density_matrix():
    """
    Test 5:
    Verify:
    - trace(rho) ≈ 1
    - rho is numerically Hermitian
    - rho is positive semi-definite (eigenvalues >= 0)
    """
    img = np.array([
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55],
        [20, 30, 40, 50, 60],
        [25, 35, 45, 55, 65],
        [30, 40, 50, 60, 70]
    ], dtype=np.float64)

    padded = apply_reflection_padding(img, pad_width=3)
    kp_y, kp_x = 2 + 3, 2 + 3  # center of 5x5 image in padded coords

    for p_name, (dy, dx) in P_OFFSETS_DICT.items():
        entropy, rho, patch = extract_single_p_entropy(padded, kp_y, kp_x, dy, dx)

        assert rho.shape == (9, 9), f"Rho for {p_name} must be 9x9"
        assert verify_density_matrix(rho), f"Rho for {p_name} failed verify_density_matrix"
        assert np.isclose(np.trace(rho), 1.0, atol=1e-8), f"Trace for {p_name} is not 1.0: {np.trace(rho)}"

        # Explicit Hermitian check: rho == rho.conj().T
        assert np.allclose(rho, rho.conj().T, atol=1e-8), f"Rho for {p_name} is not Hermitian"

        # Explicit PSD check: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-10), f"Rho for {p_name} has negative eigenvalues"


# =============================================================================
# Test 6 — Entropy Finite & Non-Negative
# =============================================================================

def test_6_entropy():
    """
    Test 6:
    Verify that entropy values are finite, non-negative within numerical tolerance,
    and bounded by theoretical maximum S_max = log2(9) ≈ 3.1699 bits.
    """
    img_flat = np.ones((20, 20), dtype=np.float64) * 128.0
    desc_flat = extract_entropy_descriptor(img_flat, (10, 10))

    s_max = np.log2(9.0)
    # A completely uniform patch has maximum entropy log2(9)
    assert np.allclose(desc_flat, s_max, atol=1e-4), (
        f"Flat image should have near-maximum entropy {s_max:.4f}, got {desc_flat}"
    )

    # Random image
    img_rand = np.random.RandomState(99).randint(0, 256, (30, 30)).astype(np.float64)
    desc_rand = extract_entropy_descriptor(img_rand, (15, 15))

    assert np.all(desc_rand >= 0.0), "Entropy must be non-negative"
    assert np.all(desc_rand <= s_max + 1e-6), f"Entropy cannot exceed log2(9) = {s_max:.4f}"
    assert np.all(np.isfinite(desc_rand)), "Entropy must be finite"


# =============================================================================
# Test 7 — Similarity & Matching Pipeline
# =============================================================================

def test_7_similarity():
    """
    Test 7:
    Run the descriptor on two test images and verify that:
    - Descriptors are generated for both images.
    - Descriptors can be matched across differing keypoint counts.
    - An image-level similarity score is produced in [0, 100]%.
    - Identical images produce ~100% similarity and near-zero descriptor distance.
    """
    # Create two test images
    img_a = np.zeros((64, 64), dtype=np.float64)
    img_a[16:48, 16:48] = 200.0  # white square

    img_b = img_a.copy()  # identical copy

    # Test identical image similarity
    res_identical = compute_image_similarity_entropy_descriptor(img_a, img_b)
    assert res_identical["num_keypoints_a"] > 0, "Image A should have keypoints"
    assert res_identical["num_keypoints_b"] > 0, "Image B should have keypoints"
    assert res_identical["num_matches"] > 0, "Should find matches between identical images"
    assert np.isclose(res_identical["mean_descriptor_distance"], 0.0, atol=1e-4), (
        f"Identical images should have near-zero distance, got {res_identical['mean_descriptor_distance']}"
    )
    assert res_identical["similarity_score"] >= 95.0, (
        f"Identical images should have >= 95% similarity, got {res_identical['similarity_score']:.2f}%"
    )

    # Test different keypoint counts matching
    desc_a = np.random.RandomState(1).rand(10, 24) * np.log2(9.0)
    desc_b = np.random.RandomState(2).rand(15, 24) * np.log2(9.0)

    match_res = match_entropy_descriptors(desc_a, desc_b, method="mutual_nn")
    assert "matches" in match_res
    assert "distance_matrix" in match_res
    assert match_res["distance_matrix"].shape == (10, 15), "Distance matrix must be (10, 15)"
    assert match_res["num_matches"] <= min(10, 15), "Mutual matches cannot exceed min(N_A, N_B)"
