"""
Unit tests for QCSI quantum information and simulation core.
"""

import numpy as np
import pytest
from qcsi.quantum import (
    amplitude_encode_pure_states,
    pure_state_fidelity,
    pure_state_trace_distance,
    compute_pairwise_quantum_metrics,
    build_ensemble_density_matrix,
    von_neumann_entropy,
    quantum_jensen_shannon_divergence,
    uhlmann_jozsa_fidelity,
    hilbert_schmidt_cosine,
    SwapTestSimulator,
)


def test_amplitude_encoding():
    raw = np.random.randn(10, 128)
    psi = amplitude_encode_pure_states(raw)
    assert psi.shape == (10, 128)
    norms = np.linalg.norm(psi, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)


def test_pure_state_fidelity_and_trace_distance():
    psi = np.random.randn(128)
    psi = psi / np.linalg.norm(psi)

    # Identical states
    fid_self = pure_state_fidelity(psi, psi)
    dist_self = pure_state_trace_distance(psi, psi)
    assert pytest.approx(fid_self, 1e-6) == 1.0
    assert pytest.approx(dist_self, 1e-6) == 0.0

    # Orthogonal states
    phi = np.random.randn(128)
    phi = phi - np.dot(phi, psi) * psi
    phi = phi / np.linalg.norm(phi)
    fid_orth = pure_state_fidelity(psi, phi)
    dist_orth = pure_state_trace_distance(psi, phi)
    assert pytest.approx(fid_orth, 1e-6) == 0.0
    assert pytest.approx(dist_orth, 1e-6) == 1.0


def test_pairwise_metrics():
    A = amplitude_encode_pure_states(np.random.randn(5, 128))
    B = amplitude_encode_pure_states(np.random.randn(7, 128))
    fid_mat, dist_mat = compute_pairwise_quantum_metrics(A, B)

    assert fid_mat.shape == (5, 7)
    assert dist_mat.shape == (5, 7)
    assert np.all(fid_mat >= 0.0) and np.all(fid_mat <= 1.0)
    assert np.all(dist_mat >= 0.0) and np.all(dist_mat <= 1.0)


def test_density_matrix_properties():
    psi = amplitude_encode_pure_states(np.random.randn(20, 128))
    weights = np.random.uniform(0.1, 1.0, size=20)
    rho = build_ensemble_density_matrix(psi, weights=weights, dim=128)

    # 1. Hermiticity
    np.testing.assert_allclose(rho, rho.T, atol=1e-10)

    # 2. Unit trace
    assert pytest.approx(float(np.trace(rho)), 1e-7) == 1.0

    # 3. Positive semi-definite (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)

    # 4. Von Neumann Entropy
    entropy = von_neumann_entropy(rho)
    assert entropy >= 0.0


def test_uhlmann_fidelity_and_qjsd():
    psi_A = amplitude_encode_pure_states(np.random.randn(15, 128))
    psi_B = amplitude_encode_pure_states(np.random.randn(15, 128))
    rho_A = build_ensemble_density_matrix(psi_A, dim=128)
    rho_B = build_ensemble_density_matrix(psi_B, dim=128)

    # Self-fidelity
    fid_self = uhlmann_jozsa_fidelity(rho_A, rho_A)
    assert pytest.approx(fid_self, 1e-4) == 1.0

    # Self-QJSD
    qjsd_self = quantum_jensen_shannon_divergence(rho_A, rho_A)
    assert pytest.approx(qjsd_self, 1e-4) == 0.0

    # Pairwise bounds
    fid_pair = uhlmann_jozsa_fidelity(rho_A, rho_B)
    qjsd_pair = quantum_jensen_shannon_divergence(rho_A, rho_B)
    hs_pair = hilbert_schmidt_cosine(rho_A, rho_B)

    assert 0.0 <= fid_pair <= 1.0
    assert 0.0 <= qjsd_pair <= 1.0
    assert 0.0 <= hs_pair <= 1.0


def test_swap_test_simulator():
    sim = SwapTestSimulator()
    psi = amplitude_encode_pure_states(np.random.randn(1, 128))[0]
    phi = amplitude_encode_pure_states(np.random.randn(1, 128))[0]

    analytic = sim.estimate_fidelity_analytic(psi, phi)
    assert 0.0 <= analytic["fidelity"] <= 1.0
    assert 0.5 <= analytic["p_zero"] <= 1.0

    sampled = sim.estimate_fidelity_sampled(psi, phi, shots=2000, seed=42)
    assert pytest.approx(sampled["fidelity_estimated"], abs=0.15) == sampled["fidelity_exact"]
