"""
QCSI Quantum Information & Simulation Core.

Implements:
1. 7-qubit Pure State Amplitude Encoding (|psi> in C^128).
2. Quantum Fidelity Kernel K = |<psi|phi>|^2 and Trace Distance D = sqrt(1 - K).
3. Mixed-State Density Operators rho = sum_i w_i |psi_i><psi_i|.
4. Von Neumann Entropy S(rho).
5. Quantum Jensen-Shannon Divergence (QJSD).
6. Uhlmann-Jozsa Mixed-State Fidelity.
7. Hilbert-Schmidt Inner Product & Cosine Similarity.
8. Simulated SWAP-Test Estimator (analytic and shot-based Qiskit simulation).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.linalg import eigh


# =============================================================================
# 1. Pure State Amplitude Encoding & Pairwise Fidelity / Trace Distance
# =============================================================================

def amplitude_encode_pure_states(descriptors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes a set of real descriptor vectors (e.g. 128-d RootSIFT) to unit norm
    in the 7-qubit Hilbert space (dim = 2^7 = 128).
    
    Args:
        descriptors: (N, 128) array of descriptors.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        (N, 128) float64 array of pure state amplitude vectors.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.empty((0, 128), dtype=np.float64)

    arr = np.asarray(descriptors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    psi = arr / norms
    return psi


def pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Computes quantum fidelity between two pure states: K = |<psi|phi>|^2.
    """
    overlap = float(np.dot(psi, phi))
    return float(np.clip(overlap ** 2, 0.0, 1.0))


def pure_state_trace_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Computes quantum trace distance between two pure states: D_tr = sqrt(1 - |<psi|phi>|^2).
    """
    fid = pure_state_fidelity(psi, phi)
    return float(np.sqrt(max(0.0, 1.0 - fid)))


def compute_pairwise_quantum_metrics(
    psi_A: np.ndarray,
    psi_B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized computation of pairwise Quantum Fidelity and Quantum Trace Distance.
    
    Args:
        psi_A: (M, 128) amplitude-encoded pure states.
        psi_B: (N, 128) amplitude-encoded pure states.
        
    Returns:
        fidelity_matrix: (M, N) where K_ij = |<psi_A[i]|psi_B[j]>|^2
        trace_dist_matrix: (M, N) where D_ij = sqrt(1 - K_ij)
    """
    if len(psi_A) == 0 or len(psi_B) == 0:
        return np.empty((len(psi_A), len(psi_B))), np.empty((len(psi_A), len(psi_B)))

    # Overlap matrix: (M, N)
    overlap = np.dot(psi_A, psi_B.T)
    fidelity_matrix = np.clip(overlap ** 2, 0.0, 1.0)
    trace_dist_matrix = np.sqrt(np.maximum(0.0, 1.0 - fidelity_matrix))
    return fidelity_matrix, trace_dist_matrix


# =============================================================================
# 2. Mixed State Density Operators & Quantum Information Metrics
# =============================================================================

def build_ensemble_density_matrix(
    psi: np.ndarray,
    weights: Optional[np.ndarray] = None,
    dim: int = 128
) -> np.ndarray:
    """
    Constructs an ensemble mixed-state density operator:
    rho = sum_i w_i |psi_i><psi_i|.
    
    Guarantees:
    - Hermitian: rho = rho^dagger
    - Positive semi-definite
    - Unit trace: Tr(rho) = 1
    
    Fallback: Maximally mixed state I / dim if no keypoints exist.
    """
    if psi is None or len(psi) == 0:
        return np.eye(dim, dtype=np.float64) / float(dim)

    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)

    if weights is None or len(weights) != n or np.sum(weights) <= 0:
        w = np.ones(n, dtype=np.float64) / float(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        total_w = np.sum(w)
        if total_w > 0:
            w = w / total_w
        else:
            w = np.ones(n, dtype=np.float64) / float(n)

    # rho = psi.T @ diag(w) @ psi
    rho = np.dot(psi.T * w, psi)

    # Symmetrize to enforce exact Hermiticity
    rho = 0.5 * (rho + rho.T)

    # Enforce unit trace
    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    else:
        rho = np.eye(dim, dtype=np.float64) / float(dim)

    return rho


def psd_matrix_sqrt(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Numerically stable matrix square root for real symmetric positive semi-definite matrix.
    sqrt(A) = V * diag(sqrt(max(lambda, 0))) * V^T.
    """
    eigvals, eigvecs = eigh(A)
    eigvals_clamped = np.maximum(eigvals, 0.0)
    sqrt_diag = np.sqrt(eigvals_clamped)
    return eigvecs @ np.diag(sqrt_diag) @ eigvecs.T


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Computes Von Neumann Entropy:
    S(rho) = - Tr(rho * log(rho)) = - sum_k lambda_k * log_base(lambda_k).
    """
    eigvals = np.linalg.eigvalsh(rho)
    pos_eig = eigvals[eigvals > 1e-15]
    if len(pos_eig) == 0:
        return 0.0
    entropy = -np.sum(pos_eig * (np.log(pos_eig) / np.log(base)))
    return float(max(0.0, entropy))


def quantum_jensen_shannon_divergence(
    rho_A: np.ndarray,
    rho_B: np.ndarray,
    base: float = 2.0
) -> float:
    """
    Computes Quantum Jensen-Shannon Divergence (QJSD):
    QJSD(rho_A, rho_B) = S((rho_A + rho_B)/2) - 0.5 * (S(rho_A) + S(rho_B)).
    
    QJSD is symmetric, bounded in [0, 1] (for base=2), and its square root is a metric.
    """
    rho_mix = 0.5 * (rho_A + rho_B)
    s_mix = von_neumann_entropy(rho_mix, base=base)
    s_a = von_neumann_entropy(rho_A, base=base)
    s_b = von_neumann_entropy(rho_B, base=base)
    qjsd = s_mix - 0.5 * (s_a + s_b)
    return float(np.clip(qjsd, 0.0, 1.0))


def uhlmann_jozsa_fidelity(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """
    Computes the Uhlmann-Jozsa quantum fidelity between two mixed states:
    F(rho_A, rho_B) = [ Tr( sqrt( sqrt(rho_A) * rho_B * sqrt(rho_A) ) ) ]^2.
    
    Computed via stable Hermitian eigenvalue decompositions.
    """
    try:
        sqrt_A = psd_matrix_sqrt(rho_A)
        M = sqrt_A @ rho_B @ sqrt_A
        M = 0.5 * (M + M.T)
        eigvals_M = np.linalg.eigvalsh(M)
        pos_eig = np.maximum(eigvals_M, 0.0)
        trace_sqrt_M = np.sum(np.sqrt(pos_eig))
        fidelity = float(trace_sqrt_M ** 2)
        return float(np.clip(fidelity, 0.0, 1.0))
    except Exception:
        # Fallback to Hilbert-Schmidt inner product if decomposition encounters numerical degeneracy
        hs = float(np.trace(rho_A @ rho_B))
        return float(np.clip(hs, 0.0, 1.0))


def hilbert_schmidt_cosine(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """
    Computes Hilbert-Schmidt cosine similarity:
    HS_cos = Tr(rho_A * rho_B) / ( ||rho_A||_F * ||rho_B||_F ).
    """
    inner = float(np.trace(rho_A @ rho_B))
    norm_A = float(np.linalg.norm(rho_A, 'fro'))
    norm_B = float(np.linalg.norm(rho_B, 'fro'))
    denom = norm_A * norm_B
    if denom <= 0:
        return 0.0
    return float(np.clip(inner / denom, 0.0, 1.0))


# =============================================================================
# 3. Simulated SWAP-Test Estimator
# =============================================================================

class SwapTestSimulator:
    """
    Simulates the quantum SWAP test circuit for measuring overlap |<psi|phi>|^2.
    Circuit:
    |0>_ancilla ── H ── • ── H ── Measure
                        │
    |psi>       ─────── X ───────
                        │
    |phi>       ─────── X ───────
    
    P(0) = (1 + |<psi|phi>|^2) / 2
    Fidelity estimator: K_hat = 2 * P(0) - 1.
    """

    def __init__(self, use_aer: bool = False):
        self.use_aer = use_aer

    def estimate_fidelity_analytic(self, psi: np.ndarray, phi: np.ndarray) -> Dict[str, float]:
        """Analytic statevector probability without shot noise."""
        fid = pure_state_fidelity(psi, phi)
        p0 = (1.0 + fid) / 2.0
        return {
            "fidelity": fid,
            "p_zero": p0,
            "estimator": 2.0 * p0 - 1.0,
        }

    def estimate_fidelity_sampled(
        self,
        psi: np.ndarray,
        phi: np.ndarray,
        shots: int = 1024,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Simulates statistical shot noise using binomial sampling of the ancilla measurement.
        """
        rng = np.random.default_rng(seed)
        fid_exact = pure_state_fidelity(psi, phi)
        p0 = (1.0 + fid_exact) / 2.0
        zeros_measured = rng.binomial(shots, p0)
        p0_hat = zeros_measured / float(shots)
        fid_hat = float(np.clip(2.0 * p0_hat - 1.0, 0.0, 1.0))

        return {
            "fidelity_exact": fid_exact,
            "fidelity_estimated": fid_hat,
            "p_zero_sampled": p0_hat,
            "shots": shots,
        }
