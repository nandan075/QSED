"""
Unit Tests for Pairwise Density Matrix and Von Neumann Entropy Calculations.

Validation checks:
- Check 1: Identical normalized quantum states produce zero pairwise entropy within numerical tolerance.
- Check 2: Orthogonal normalized quantum states produce one bit of pairwise entropy.
- Check 3: The entropy is symmetric: S_{ij} = S_{ji}.
- Check 4: The density matrix is Hermitian, positive semidefinite within numerical tolerance, and has trace one.
- Check 5: Pairwise entropy is bounded between zero and one bit: 0 <= S_{ij} <= 1.
- Check 6: The optimized and reference entropy calculations agree within numerical tolerance.
"""

import pytest
import numpy as np

from pairwise_von_neumann_entropy.quantum_state_encoder import normalize_quantum_pure_state
from pairwise_von_neumann_entropy.pairwise_entropy import (
    construct_pairwise_density_matrix,
    verify_density_matrix,
    compute_pairwise_entropy_reference,
    compute_pairwise_entropy_optimized,
    compute_pairwise_entropy_matrix_optimized,
    compute_pairwise_entropy_matrix_reference,
)


def test_identical_states_produce_zero_entropy():
    """Check 1: Identical normalized quantum states produce S_ij = 0.0 bits."""
    rng = np.random.RandomState(42)
    dim = 81

    for _ in range(5):
        raw_vec = rng.randn(dim)
        psi = normalize_quantum_pure_state(raw_vec)

        # Reference density matrix formulation
        s_ref = compute_pairwise_entropy_reference(psi, psi)
        assert np.isclose(s_ref, 0.0, atol=1e-7), f"Expected 0.0 for identical states, got {s_ref}"

        # Optimized subspace eigenvalue formulation
        s_opt = compute_pairwise_entropy_optimized(psi, psi)
        assert np.isclose(s_opt, 0.0, atol=1e-7), f"Expected 0.0 for identical states, got {s_opt}"


def test_orthogonal_states_produce_one_bit_entropy():
    """Check 2: Orthogonal normalized quantum states produce S_ij = 1.0 bit."""
    dim = 81

    # Canonical standard basis states |0> and |1>
    psi_0 = np.zeros(dim, dtype=np.float64)
    psi_0[0] = 1.0

    psi_1 = np.zeros(dim, dtype=np.float64)
    psi_1[1] = 1.0

    assert np.isclose(np.dot(psi_0, psi_1), 0.0, atol=1e-12), "States must be strictly orthogonal"

    s_ref = compute_pairwise_entropy_reference(psi_0, psi_1)
    assert np.isclose(s_ref, 1.0, atol=1e-7), f"Expected 1.0 bit for orthogonal states, got {s_ref}"

    s_opt = compute_pairwise_entropy_optimized(psi_0, psi_1)
    assert np.isclose(s_opt, 1.0, atol=1e-7), f"Expected 1.0 bit for orthogonal states, got {s_opt}"


def test_entropy_symmetry():
    """Check 3: The entropy is strictly symmetric: S_{ij} = S_{ji}."""
    rng = np.random.RandomState(123)
    dim = 64

    for _ in range(10):
        psi_i = normalize_quantum_pure_state(rng.randn(dim))
        psi_j = normalize_quantum_pure_state(rng.randn(dim))

        s_ij = compute_pairwise_entropy_optimized(psi_i, psi_j)
        s_ji = compute_pairwise_entropy_optimized(psi_j, psi_i)
        assert np.isclose(s_ij, s_ji, atol=1e-10), f"Symmetry violated: S_ij={s_ij} != S_ji={s_ji}"

        s_ref_ij = compute_pairwise_entropy_reference(psi_i, psi_j)
        s_ref_ji = compute_pairwise_entropy_reference(psi_j, psi_i)
        assert np.isclose(s_ref_ij, s_ref_ji, atol=1e-10), f"Reference symmetry violated"


def test_density_matrix_physical_validity():
    """Check 4: Density matrix is Hermitian, PSD, and has trace exactly 1.0."""
    rng = np.random.RandomState(456)
    dim = 16

    for _ in range(10):
        psi_i = normalize_quantum_pure_state(rng.randn(dim))
        psi_j = normalize_quantum_pure_state(rng.randn(dim))

        rho = construct_pairwise_density_matrix(psi_i, psi_j)

        is_valid, diag = verify_density_matrix(rho, tol=1e-8)
        assert is_valid, f"Density matrix validation failed: {diag}"
        assert diag["is_hermitian"], "rho must be Hermitian"
        assert diag["is_trace_one"], f"Tr(rho) must be 1.0, got {diag['trace']}"
        assert diag["is_positive_semidefinite"], f"Min eigenvalue must be >= 0, got {diag['min_eigenvalue']}"


def test_entropy_bounded_zero_to_one():
    """Check 5: Pairwise entropy is strictly bounded: 0.0 <= S_{ij} <= 1.0."""
    rng = np.random.RandomState(789)
    dim = 32

    for _ in range(25):
        psi_i = normalize_quantum_pure_state(rng.randn(dim))
        psi_j = normalize_quantum_pure_state(rng.randn(dim))

        s = compute_pairwise_entropy_optimized(psi_i, psi_j)
        assert -1e-10 <= s <= 1.0 + 1e-10, f"Entropy out of [0, 1] range: {s}"


def test_optimized_and_reference_calculations_agree():
    """Check 6: Subspace eigenvalue optimization agrees with density matrix diagonalization within 1e-7."""
    rng = np.random.RandomState(101)
    dim = 25
    n = 6

    states = np.array([normalize_quantum_pure_state(rng.randn(dim)) for _ in range(n)])

    # Compute full matrices using both methods
    mat_opt = compute_pairwise_entropy_matrix_optimized(states)
    mat_ref = compute_pairwise_entropy_matrix_reference(states)

    assert mat_opt.shape == (n, n)
    assert mat_ref.shape == (n, n)
    assert np.allclose(mat_opt, mat_ref, atol=1e-7), (
        f"Optimized and reference matrices disagree!\nMax diff: {np.max(np.abs(mat_opt - mat_ref))}"
    )
