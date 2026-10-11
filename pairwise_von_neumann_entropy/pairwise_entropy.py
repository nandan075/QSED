"""
Pairwise Von Neumann Entropy Module.

Mathematical Formulation:
For any unordered pair of keypoints (k_i, k_j) with normalized pure state vectors
|psi_i>, |psi_j> in H_d (with <psi_i|psi_i> = <psi_j|psi_j> = 1.0), the statistical
ensemble density matrix is an equal mixture:

    rho_{ij} = 1/2 ( |psi_i><psi_i| + |psi_j><psi_j| )

Properties of rho_{ij}:
1. Hermitian: rho_{ij}^dagger = rho_{ij}.
2. Positive semi-definite: all eigenvalues lambda >= 0.
3. Unit trace: Tr(rho_{ij}) = 1/2 ( <psi_i|psi_i> + <psi_j|psi_j> ) = 1.0.

Von Neumann Entropy:
    S_{ij} = - Tr( rho_{ij} log_2 rho_{ij} ) = - sum_{lambda_k > 0} lambda_k log_2 lambda_k

Subspace Eigenvalue Optimization:
Since rho_{ij} is formed by a mixture of two pure states, its range is confined to at most
a 2-dimensional subspace spanned by {|psi_i>, |psi_j>}. All other (d - 2) eigenvalues are identically zero!
Let the transition probability (quantum fidelity) be:
    c_{ij} = |<psi_i|psi_j>|^2 in [0, 1].

The two potentially non-zero eigenvalues are analytically given by:
    lambda_{pm} = (1 pm sqrt(c_{ij})) / 2

Proof:
    Tr(rho_{ij}) = lambda_+ + lambda_- = 1.
    Tr(rho_{ij}^2) = lambda_+^2 + lambda_-^2 = (1 + c_{ij}) / 2.
    (lambda_+ - lambda_-)^2 = 2 Tr(rho_{ij}^2) - (Tr(rho_{ij}))^2 = 1 + c_{ij} - 1 = c_{ij}.
    => lambda_+ - lambda_- = sqrt(c_{ij}).
    => lambda_{pm} = (1 pm sqrt(c_{ij})) / 2.

Hence:
    S_{ij} = - lambda_+ log_2(lambda_+) - lambda_- log_2(lambda_-)

Extreme Values:
- Identical states (|psi_i> = |psi_j>): c_{ij} = 1 => lambda_+ = 1, lambda_- = 0 => S_{ij} = 0.0 bits.
- Orthogonal states (<psi_i|psi_j> = 0): c_{ij} = 0 => lambda_+ = 1/2, lambda_- = 1/2 => S_{ij} = 1.0 bit.
- Bounded Range: 0.0 <= S_{ij} <= 1.0 bit for all normalized pure states.

Complexity:
- Reference diagonalization: O(n^2 * d^3) floating-point operations.
- Optimized overlap calculation: O(n^2 * d) via matrix multiplication, avoiding density matrix allocation.
"""

from typing import Tuple, Optional, Union
import numpy as np


def construct_pairwise_density_matrix(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
) -> np.ndarray:
    """
    Construct the pairwise mixed-state density matrix:
        rho = 1/2 ( |psi_i><psi_i| + |psi_j><psi_j| )

    Args:
        psi_i: 1D normalized pure state vector of dimension d.
        psi_j: 1D normalized pure state vector of dimension d.

    Returns:
        2D float64 array of shape (d, d) representing the density operator rho.
    """
    pi_i = np.asarray(psi_i, dtype=np.float64).flatten()
    pi_j = np.asarray(psi_j, dtype=np.float64).flatten()

    # Outer products: |psi><psi|
    rho_i = np.outer(pi_i, pi_i.conj())
    rho_j = np.outer(pi_j, pi_j.conj())

    rho = 0.5 * (rho_i + rho_j)

    # Enforce exact numerical Hermiticity
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def verify_density_matrix(
    rho: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[bool, dict]:
    """
    Numerically verify that rho is a physically valid quantum density operator.

    Checks:
    1. Hermiticity: rho == rho^dagger
    2. Unit Trace: |Tr(rho) - 1.0| < tol
    3. Positive Semi-Definiteness: all eigenvalues lambda >= -tol

    Args:
        rho: Square matrix of shape (d, d).
        tol: Tolerance for floating-point comparisons.

    Returns:
        Tuple of (is_valid, diagnostics_dict).
    """
    d = rho.shape[0]
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace_val = float(np.real(np.trace(rho)))
    is_trace_one = bool(np.isclose(trace_val, 1.0, atol=tol))

    eigvals = np.linalg.eigvalsh(rho)
    min_eig = float(np.min(eigvals))
    is_psd = bool(min_eig >= -tol)

    is_valid = is_hermitian and is_trace_one and is_psd

    diagnostics = {
        "dimension": d,
        "is_hermitian": is_hermitian,
        "trace": trace_val,
        "is_trace_one": is_trace_one,
        "min_eigenvalue": min_eig,
        "is_positive_semidefinite": is_psd,
        "all_eigenvalues": eigvals,
    }
    return is_valid, diagnostics


def compute_pairwise_entropy_reference(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    log_base: float = 2.0,
    tol: float = 1e-15,
) -> float:
    """
    Reference calculation of Von Neumann entropy S(rho_{ij}) by explicit density
    matrix construction and full Hermitian eigendecomposition.

    Args:
        psi_i: 1D normalized pure state vector.
        psi_j: 1D normalized pure state vector.
        log_base: Logarithm base (default: 2.0 for bits).
        tol: Epsilon cutoff below which eigenvalues are considered zero.

    Returns:
        Von Neumann entropy S_{ij} in [0.0, 1.0] bits.
    """
    rho = construct_pairwise_density_matrix(psi_i, psi_j)
    eigvals = np.linalg.eigvalsh(rho)

    # Filter strictly positive eigenvalues above tolerance
    pos_eig = eigvals[eigvals > tol]
    if len(pos_eig) == 0:
        return 0.0

    entropy = -np.sum(pos_eig * (np.log(pos_eig) / np.log(log_base)))
    return float(np.clip(entropy, 0.0, 1.0))


def compute_pairwise_entropy_optimized(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    log_base: float = 2.0,
    eps: float = 1e-15,
) -> float:
    """
    Analytical, optimized calculation of Von Neumann entropy S_{ij}.
    Avoids allocating or diagonalizing the d x d density matrix.

    Uses the 2D subspace eigenvalue formula:
        c_{ij} = |<psi_i|psi_j>|^2
        lambda_{pm} = (1 pm sqrt(c_{ij})) / 2
        S_{ij} = - sum_{lambda in {lambda_+, lambda_-}} lambda log_2(lambda)

    Args:
        psi_i: 1D normalized pure state vector.
        psi_j: 1D normalized pure state vector.
        log_base: Logarithm base (default: 2.0).
        eps: Tolerance for numerical zero.

    Returns:
        Entropy in bits in [0.0, 1.0].
    """
    # Overlap / Transition amplitude
    overlap = float(np.dot(psi_i, psi_j))
    c_ij = float(np.clip(overlap ** 2, 0.0, 1.0))

    sqrt_c = np.sqrt(c_ij)
    lam_plus = 0.5 * (1.0 + sqrt_c)
    lam_minus = 0.5 * (1.0 - sqrt_c)

    entropy = 0.0
    if lam_plus > eps:
        entropy -= lam_plus * (np.log(lam_plus) / np.log(log_base))
    if lam_minus > eps:
        entropy -= lam_minus * (np.log(lam_minus) / np.log(log_base))

    return float(np.clip(entropy, 0.0, 1.0))


def compute_pairwise_entropy_matrix_optimized(
    states: np.ndarray,
    log_base: float = 2.0,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Vectorized computation of the entire symmetric pairwise entropy matrix E in R^{n x n}.
    Achieves O(n^2 * d) complexity via matrix multiplication:
        G = states @ states.T       (Gram matrix of inner products)
        C = clip(G^2, 0.0, 1.0)     (transition probabilities c_{ij})
        Lambda_{pm} = (1 pm sqrt(C)) / 2
        E_{ij} = - Lambda_+ log_2(Lambda_+) - Lambda_- log_2(Lambda_-)
        E_{ii} = 0.0                (storage convention)

    Args:
        states: 2D array of shape (n, d) containing n normalized pure state vectors.
        log_base: Logarithm base (default: 2.0).
        eps: Small tolerance for numerical zero.

    Returns:
        Symmetric array E of shape (n, n) with zeros on the diagonal.
    """
    n = states.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    # Gram matrix of overlaps: G[i, j] = <psi_i | psi_j>
    gram = np.dot(states, states.T)
    c_mat = np.clip(gram ** 2, 0.0, 1.0)

    sqrt_c = np.sqrt(c_mat)
    lam_plus = 0.5 * (1.0 + sqrt_c)
    lam_minus = 0.5 * (1.0 - sqrt_c)

    log_div = np.log(log_base)

    # Safe log computation (lim x->0: x log x = 0)
    safe_plus = np.where(lam_plus > eps, lam_plus, 1.0)
    safe_minus = np.where(lam_minus > eps, lam_minus, 1.0)

    term_plus = np.where(lam_plus > eps, lam_plus * (np.log(safe_plus) / log_div), 0.0)
    term_minus = np.where(lam_minus > eps, lam_minus * (np.log(safe_minus) / log_div), 0.0)

    entropy_matrix = -(term_plus + term_minus)
    entropy_matrix = np.clip(entropy_matrix, 0.0, 1.0)

    # Enforce exact symmetry: E = 1/2 (E + E.T)
    entropy_matrix = 0.5 * (entropy_matrix + entropy_matrix.T)

    # Zero out diagonal: E_{ii} = 0 (storage convention for keypoint relationship)
    np.fill_diagonal(entropy_matrix, 0.0)

    return entropy_matrix


def compute_pairwise_entropy_matrix_reference(
    states: np.ndarray,
    log_base: float = 2.0,
    tol: float = 1e-15,
) -> np.ndarray:
    """
    Reference pairwise entropy matrix computation using explicit density matrix
    diagonalization for all n(n-1)/2 pairs.

    Args:
        states: 2D array of shape (n, d).
        log_base: Logarithm base.
        tol: Eigenvalue tolerance.

    Returns:
        Symmetric array E of shape (n, n) with zeros on the diagonal.
    """
    n = states.shape[0]
    e_mat = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return e_mat

    for i in range(n):
        for j in range(i + 1, n):
            s_ij = compute_pairwise_entropy_reference(states[i], states[j], log_base=log_base, tol=tol)
            e_mat[i, j] = s_ij
            e_mat[j, i] = s_ij

    return e_mat


def compute_pairwise_entropy_matrix(
    states: np.ndarray,
    method: str = "optimized",
    log_base: float = 2.0,
) -> np.ndarray:
    """
    Unified entry point for computing the symmetric pairwise entropy matrix E.

    Args:
        states: (n, d) array of normalized pure state vectors.
        method: 'optimized' (default, analytical subspace eigenvalues) or 'reference' (density matrix diagonalization).
        log_base: Logarithm base (default: 2.0).

    Returns:
        (n, n) float64 symmetric matrix with zeros on diagonal.
    """
    if method == "optimized":
        return compute_pairwise_entropy_matrix_optimized(states, log_base=log_base)
    elif method == "reference":
        return compute_pairwise_entropy_matrix_reference(states, log_base=log_base)
    else:
        raise ValueError(f"Unknown pairwise entropy method: '{method}'. Choose 'optimized' or 'reference'.")
