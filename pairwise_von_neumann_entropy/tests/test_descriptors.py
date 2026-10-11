"""
Unit Tests for Pairwise Entropy Matrix and (n - 1)-Dimensional Keypoint Descriptors.

Validation checks:
- Check 1: A symmetric entropy matrix E is generated correctly (E == E^T).
- Check 2: The diagonal is exactly zero (E_{ii} == 0.0) as an algorithm storage convention.
- Check 3: Every keypoint descriptor contains exactly (n - 1) entries.
- Check 4: Boundary cases with n = 0 (empty) and n = 1 (single keypoint) are handled safely.
- Check 5: Preserves coordinates, detector metadata, and descriptor mapping accurately.
"""

import pytest
import numpy as np

from pairwise_von_neumann_entropy.quantum_state_encoder import normalize_quantum_pure_state
from pairwise_von_neumann_entropy.descriptor_builder import (
    build_keypoint_descriptors,
    KeypointPairwiseDescriptor,
    ImageEntropyRepresentation,
)


def test_symmetric_entropy_matrix_and_zero_diagonal():
    """Checks 1 & 2: E is symmetric (E == E^T) and diagonal elements are 0.0."""
    rng = np.random.RandomState(42)
    n = 12
    dim = 49

    states = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n)])
    kps_xy = [(float(i * 10), float(i * 10)) for i in range(n)]

    rep = build_keypoint_descriptors(states, kps_xy, image_shape=(200, 200))
    E = rep.entropy_matrix

    assert E.shape == (n, n), f"Expected shape ({n}, {n}), got {E.shape}"
    assert np.allclose(E, E.T, atol=1e-12), "Entropy matrix E must be strictly symmetric"
    assert np.allclose(np.diag(E), 0.0, atol=1e-12), "Diagonal of E must be exactly 0.0"


def test_descriptor_length_is_n_minus_one():
    """Check 3: Each keypoint descriptor contains exactly (n - 1) entries."""
    rng = np.random.RandomState(77)
    for n in [2, 5, 10, 25]:
        dim = 16
        states = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n)])
        kps_xy = [(float(i * 5), float(i * 5)) for i in range(n)]

        rep = build_keypoint_descriptors(states, kps_xy)

        assert len(rep.descriptors) == n, f"Expected {n} descriptors"
        for i, desc_obj in enumerate(rep.descriptors):
            assert desc_obj.descriptor.shape == (n - 1,), (
                f"Descriptor for keypoint {i} has shape {desc_obj.descriptor.shape}, expected ({n - 1},)"
            )
            # Verify descriptor matches row of E excluding E[i, i]
            expected_vec = np.concatenate([rep.entropy_matrix[i, :i], rep.entropy_matrix[i, i + 1 :]])
            assert np.allclose(desc_obj.descriptor, expected_vec), (
                f"Descriptor content mismatch for keypoint {i}"
            )


def test_edge_cases_empty_and_single_keypoint():
    """Check 4: Empty (n = 0) and single keypoint (n = 1) cases do not crash."""
    dim = 64

    # 1. Empty Case: n = 0
    empty_states = np.empty((0, dim), dtype=np.float64)
    rep_empty = build_keypoint_descriptors(empty_states, [])
    assert rep_empty.num_keypoints == 0
    assert rep_empty.entropy_matrix.shape == (0, 0)
    assert len(rep_empty.descriptors) == 0

    # 2. Single Keypoint Case: n = 1
    single_state = normalize_quantum_pure_state(np.ones(dim))
    states_1 = single_state.reshape(1, -1)
    kps_1 = [(50.0, 50.0)]
    meta_1 = [{"response": 1.0, "scale": 2.0}]

    rep_single = build_keypoint_descriptors(states_1, kps_1, metadata_list=meta_1)
    assert rep_single.num_keypoints == 1
    assert rep_single.entropy_matrix.shape == (1, 1)
    assert rep_single.entropy_matrix[0, 0] == 0.0

    desc_0 = rep_single.descriptors[0]
    # For n = 1, (n - 1) = 0, so descriptor should be an empty 0-dimensional array
    assert desc_0.descriptor.shape == (0,)
    assert desc_0.x == 50.0
    assert desc_0.y == 50.0
    assert desc_0.metadata["scale"] == 2.0


def test_metadata_and_coordinate_preservation():
    """Check 5: Preserves coordinates and detector metadata accurately."""
    rng = np.random.RandomState(88)
    n = 4
    dim = 16

    states = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n)])
    kps_xy = [(12.5, 34.5), (67.0, 89.0), (101.2, 203.4), (150.0, 150.0)]
    meta = [
        {"response": 0.85, "scale": 1.2, "detector": "sift"},
        {"response": 0.42, "scale": 2.0, "detector": "sift"},
        {"response": 0.99, "scale": 0.8, "detector": "sift"},
        {"response": 0.15, "scale": 1.5, "detector": "sift"},
    ]

    rep = build_keypoint_descriptors(states, kps_xy, metadata_list=meta)

    for i in range(n):
        desc = rep.descriptors[i]
        assert desc.index == i
        assert desc.pt == kps_xy[i]
        assert desc.metadata["response"] == meta[i]["response"]
        assert desc.metadata["scale"] == meta[i]["scale"]
        assert np.allclose(desc.quantum_state, states[i])
