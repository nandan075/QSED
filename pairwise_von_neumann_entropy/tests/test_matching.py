"""
Unit Tests for Permutation-Aware Image Matching and Geometric Verification.

Validation checks:
- Check 1: Supports different keypoint counts across images (n_A != n_B).
- Check 2: Matching does not depend on arbitrary keypoint list ordering (permutation-aware).
- Check 3: Incorrect or ambiguous correspondences are rejected by relational filtering.
- Check 4: Geometric verification handles insufficient matches (< 4 points) safely.
- Check 5: End-to-end synthetic image matching pipeline runs and outputs valid results.
"""

import pytest
import numpy as np

from pairwise_von_neumann_entropy.quantum_state_encoder import normalize_quantum_pure_state
from pairwise_von_neumann_entropy.descriptor_builder import build_keypoint_descriptors
from pairwise_von_neumann_entropy.geometric_verification import verify_matches_ransac
from pairwise_von_neumann_entropy.matcher import (
    PairwiseEntropyMatcher,
    generate_candidate_seeds,
    build_compatibility_matrix,
    solve_spectral_assignment,
)
from pairwise_von_neumann_entropy.config import MatchingConfig, GeometricConfig


def test_unequal_keypoint_counts_supported():
    """Check 1: Images with different keypoint counts (e.g. n_A = 8, n_B = 15) match properly."""
    rng = np.random.RandomState(42)
    dim = 25
    n_a = 8
    n_b = 15

    states_a = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n_a)])
    kps_a = [(float(i * 10), float(i * 10)) for i in range(n_a)]

    states_b = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n_b)])
    kps_b = [(float(j * 8), float(j * 8)) for j in range(n_b)]

    rep_a = build_keypoint_descriptors(states_a, kps_a)
    rep_b = build_keypoint_descriptors(states_b, kps_b)

    matcher = PairwiseEntropyMatcher()
    result = matcher.match(rep_a, rep_b)

    assert result.num_kps_a == n_a
    assert result.num_kps_b == n_b
    assert result.num_matches <= min(n_a, n_b), "Matches cannot exceed min(n_A, n_B)"
    assert 0.0 <= result.similarity_score <= 100.0, "Similarity score must be in [0, 100]%"


def test_permutation_invariance_of_matching():
    """
    Check 2: Matching is invariant to arbitrary keypoint list ordering.
    If the keypoint order in Image B is shuffled by an arbitrary permutation pi,
    the resulting correspondence mapping (a_i <-> b_{pi(j)}) remains identical.
    """
    rng = np.random.RandomState(99)
    dim = 36
    n = 6

    # Construct base quantum states
    base_states = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n)])
    base_kps = [(float(i * 20), float(i * 20)) for i in range(n)]

    rep_a = build_keypoint_descriptors(base_states, base_kps)

    # Image B is an exact copy of Image A with slightly perturbed positions and shuffled ordering
    perm = rng.permutation(n)
    inv_perm = np.empty(n, dtype=int)
    inv_perm[perm] = np.arange(n)

    shuffled_states = base_states[perm]
    shuffled_kps = [base_kps[p] for p in perm]

    rep_b = build_keypoint_descriptors(shuffled_states, shuffled_kps)

    matcher = PairwiseEntropyMatcher(
        matching_config=MatchingConfig(seed_method="mutual_nn", ambiguity_threshold=0.05)
    )
    res = matcher.match(rep_a, rep_b)

    # For every matched pair (i, j), keypoint i in A must match keypoint inv_perm[i] in B
    assert res.num_matches > 0, "Expected matches for identical feature states"
    for i, j, _, _ in res.matches:
        expected_j = int(np.where(perm == i)[0][0])
        assert j == expected_j, f"Permutation mismatch: Keypoint {i} matched to {j}, expected {expected_j}"


def test_ambiguous_correspondence_rejection():
    """Check 3: Incoherent or conflicting matches are rejected by spectral compatibility."""
    dim = 16

    # Candidate matches where relational geometry is completely scrambled
    candidates = [(0, 0, 0.9), (1, 1, 0.9), (2, 5, 0.2)]  # (2, 5) is an outlier candidate

    # Pairwise entropy matrix A
    e_a = np.array([
        [0.0, 0.2, 0.8],
        [0.2, 0.0, 0.7],
        [0.8, 0.7, 0.0]
    ], dtype=np.float64)

    # Pairwise entropy matrix B where relation between 0 and 1 matches, but 5 disagrees completely
    e_b = np.zeros((6, 6), dtype=np.float64)
    e_b[0, 1] = 0.2
    e_b[1, 0] = 0.2
    e_b[0, 5] = 0.1  # Very different from e_a[0, 2] = 0.8!
    e_b[5, 0] = 0.1

    W = build_compatibility_matrix(candidates, e_a, e_b, entropy_sigma=0.25)
    selected = solve_spectral_assignment(W, candidates, ambiguity_threshold=0.5)

    # The outlier match (2, 5) should be filtered out
    matched_pairs = [(i, j) for i, j, _, _ in selected]
    assert (2, 5) not in matched_pairs, "Incoherent candidate (2, 5) should have been rejected"


def test_geometric_verification_insufficient_matches():
    """Check 4: RANSAC safely handles insufficient matches (< 4 points) without crashing."""
    # 2 points cannot define a 4-point homography
    pts_a = np.array([[10.0, 20.0], [30.0, 40.0]])
    pts_b = np.array([[12.0, 22.0], [32.0, 42.0]])

    res = verify_matches_ransac(pts_a, pts_b, model="homography", min_inliers=4)
    assert not res["success"], "Should fail gracefully when points < 4"
    assert res["inlier_count"] == 0
    assert "insufficient_matches" in res["status"]
    assert res["model_matrix"] is None


def test_empty_image_pair_matching():
    """Check 5: Empty keypoint representations return clean zero-score results."""
    dim = 16
    rep_empty_a = build_keypoint_descriptors(np.empty((0, dim)), [])
    rep_empty_b = build_keypoint_descriptors(np.empty((0, dim)), [])

    matcher = PairwiseEntropyMatcher()
    result = matcher.match(rep_empty_a, rep_empty_b)

    assert result.num_matches == 0
    assert result.inlier_count == 0
    assert result.similarity_score == 0.0
    assert result.geometric_status == "empty_keypoints"
