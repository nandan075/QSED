"""
Permutation-Aware Relational Keypoint Matcher Module.

Mathematical Problem:
Given Image A with n_A keypoints and Image B with n_B keypoints (where n_A != n_B in general):
- The pairwise entropy matrix E^A has shape (n_A, n_A).
- The pairwise entropy matrix E^B has shape (n_B, n_B).
- Individual descriptor vectors D(k_i^A) in R^{n_A - 1} and D(k_j^B) in R^{n_B - 1} have
  DIFFERENT dimensions and unordered indices. They cannot be compared directly by index.

Permutation-Aware Matching Strategy:
1. Candidate Correspondence Generation (Seed Hypotheses):
   Compute appearance overlap between quantum states |psi_i^A> and |psi_j^B>:
       A(i, j) = |<psi_i^A | psi_j^B>|^2 in [0, 1]  (Quantum Fidelity)
   Filter candidate matches via Mutual Nearest Neighbors, Lowe's Ratio Test, or Top-K.
   Produces a candidate list M = { m_1, m_2, ..., m_K }, where m = (i, j).

2. Relational Pairwise Entropy Compatibility:
   For two candidate matches m = (i, j) and m' = (p, q) (with i != p and j != q),
   the difference between the intra-image pairwise entropy relations is:
       Delta E(m, m') = | E^A_{ip} - E^B_{jq} |
   Compatibility affinity:
       W_{m, m'} = exp( - Delta E(m, m') / sigma_E ) * sqrt( A(i, j) * A(p, q) )
   If i == p or j == q (one-to-many conflict), W_{m, m'} = 0.0.
   Diagonal entry: W_{m, m} = A(i, j).

3. Spectral Graph Matching & Conflict Resolution:
   The compatibility matrix W in R^{K x K} encodes a high-order relational agreement graph.
   We compute the principal eigenvector of W via power iteration:
       x_{t+1} = (W x_t) / ||W x_t||_2
   The elements of the principal eigenvector x^* score the mutual coherence of each candidate match.
   A greedy discretization selects mutually coherent matches satisfying the strict 1-to-1 constraint.

4. Ambiguity Filtering & RANSAC Verification:
   Filter matches falling below the ambiguity threshold.
   Run geometric verification (Homography/Fundamental RANSAC) to yield the final verified correspondences.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .descriptor_builder import ImageEntropyRepresentation, KeypointPairwiseDescriptor
from .geometric_verification import verify_matches_ransac
from .config import MatchingConfig, GeometricConfig


@dataclass
class MatchResult:
    """
    Complete results container for two-image pairwise entropy matching.

    Attributes:
        matches: List of tuples (idx_a, idx_b, compatibility_score, appearance_fidelity).
        pts_a: (M, 2) matched coordinates in Image A.
        pts_b: (M, 2) matched coordinates in Image B.
        inlier_mask: Boolean array indicating RANSAC inliers.
        inlier_count: Total number of geometrically verified inliers.
        similarity_score: Image-level similarity score in [0.0, 100.0]%.
        mean_entropy_error: Mean relational entropy error |E_ip^A - E_jq^B| across matched edges.
        geometric_status: Status string from RANSAC verification.
        model_matrix: Estimated 3x3 homography or fundamental matrix.
        num_kps_a: Total keypoints in Image A.
        num_kps_b: Total keypoints in Image B.
    """
    matches: List[Tuple[int, int, float, float]]
    pts_a: np.ndarray
    pts_b: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    similarity_score: float
    mean_entropy_error: float
    geometric_status: str
    model_matrix: Optional[np.ndarray]
    num_kps_a: int
    num_kps_b: int

    @property
    def num_matches(self) -> int:
        return len(self.matches)


def generate_candidate_seeds(
    states_a: np.ndarray,
    states_b: np.ndarray,
    seed_method: str = "mutual_nn",
    ratio_threshold: float = 0.8,
    top_k: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Generate initial candidate correspondences based on quantum pure state fidelity:
        F_{ij} = |<psi_i^A | psi_j^B>|^2 in [0.0, 1.0].

    Args:
        states_a: (n_A, d) normalized quantum states of Image A.
        states_b: (n_B, d) normalized quantum states of Image B.
        seed_method: 'mutual_nn', 'ratio_test', or 'top_k'.
        ratio_threshold: Ratio test cutoff.
        top_k: Candidate count per keypoint in top_k mode.

    Returns:
        List of candidate tuples (idx_a, idx_b, fidelity).
    """
    n_a = len(states_a)
    n_b = len(states_b)
    if n_a == 0 or n_b == 0:
        return []

    # Pairwise overlap matrix G = states_a @ states_b.T
    overlap = np.dot(states_a, states_b.T)
    fidelity_mat = np.clip(overlap ** 2, 0.0, 1.0)
    # Quantum distance D = sqrt(1 - Fidelity)
    dist_mat = np.sqrt(np.maximum(0.0, 1.0 - fidelity_mat))

    candidates: List[Tuple[int, int, float]] = []

    if seed_method == "mutual_nn":
        best_b_for_a = np.argmin(dist_mat, axis=1)
        best_a_for_b = np.argmin(dist_mat, axis=0)

        for i in range(n_a):
            j = int(best_b_for_a[i])
            if best_a_for_b[j] == i:
                candidates.append((i, j, float(fidelity_mat[i, j])))

    elif seed_method == "ratio_test":
        for i in range(n_a):
            dists = dist_mat[i, :]
            sorted_idx = np.argsort(dists)
            if len(sorted_idx) >= 2:
                d1 = dists[sorted_idx[0]]
                d2 = dists[sorted_idx[1]]
                if d1 < ratio_threshold * d2:
                    j = int(sorted_idx[0])
                    candidates.append((i, j, float(fidelity_mat[i, j])))
            elif len(sorted_idx) == 1:
                candidates.append((i, int(sorted_idx[0]), float(fidelity_mat[i, sorted_idx[0]])))

    elif seed_method == "top_k":
        for i in range(n_a):
            sorted_j = np.argsort(dist_mat[i, :])[: min(top_k, n_b)]
            for j in sorted_j:
                candidates.append((i, int(j), float(fidelity_mat[i, j])))

    else:
        raise ValueError(f"Unknown seed generation method: '{seed_method}'")

    return candidates


def build_compatibility_matrix(
    candidates: List[Tuple[int, int, float]],
    entropy_matrix_a: np.ndarray,
    entropy_matrix_b: np.ndarray,
    entropy_sigma: float = 0.25,
    relational_weight: float = 0.6,
) -> np.ndarray:
    """
    Build the second-order relational compatibility affinity matrix W in R^{K x K}.

    For candidate match m = (i, j) and m' = (p, q):
    - If i == p or j == q (1-to-1 conflict): W_{m, m'} = 0.0.
    - If i != p and j != q:
          diff = | E^A_{ip} - E^B_{jq} |
          rel_affinity = exp( - diff / entropy_sigma )
          app_affinity = sqrt( A(i, j) * A(p, q) )
          W_{m, m'} = relational_weight * rel_affinity + (1 - relational_weight) * app_affinity
    - If m == m': W_{m, m} = A(i, j).

    Args:
        candidates: List of K candidate matches (i, j, fidelity).
        entropy_matrix_a: (n_A, n_A) pairwise entropy matrix of Image A.
        entropy_matrix_b: (n_B, n_B) pairwise entropy matrix of Image B.
        entropy_sigma: Bandwidth parameter for relational difference.
        relational_weight: Weight balance in [0, 1].

    Returns:
        Symmetric non-negative matrix W of shape (K, K).
    """
    k = len(candidates)
    W = np.zeros((k, k), dtype=np.float64)
    if k == 0:
        return W

    for u in range(k):
        i, j, fid_u = candidates[u]
        W[u, u] = fid_u  # Diagonal self-affinity

        for v in range(u + 1, k):
            p, q, fid_v = candidates[v]

            # Structural constraint: strict 1-to-1 mapping
            if i == p or j == q:
                continue

            # Relational pairwise entropy difference
            delta_e = abs(float(entropy_matrix_a[i, p]) - float(entropy_matrix_b[j, q]))
            rel_score = float(np.exp(-delta_e / entropy_sigma))
            app_score = float(np.sqrt(fid_u * fid_v))

            score = relational_weight * rel_score + (1.0 - relational_weight) * app_score
            W[u, v] = score
            W[v, u] = score

    return W


def solve_spectral_assignment(
    W: np.ndarray,
    candidates: List[Tuple[int, int, float]],
    num_iterations: int = 30,
    ambiguity_threshold: float = 0.15,
) -> List[Tuple[int, int, float, float]]:
    """
    Solve the relational correspondence problem using the spectral matching method
    (Leordeanu & Hebert) with greedy conflict resolution.

    Args:
        W: (K, K) affinity matrix.
        candidates: List of K candidate tuples (i, j, fidelity).
        num_iterations: Number of power iteration steps.
        ambiguity_threshold: Cutoff on eigenvector scores.

    Returns:
        List of selected matches: (idx_a, idx_b, compatibility_score, fidelity).
    """
    k = len(candidates)
    if k == 0:
        return []

    # Power iteration to find principal eigenvector
    x = np.ones(k, dtype=np.float64) / np.sqrt(k)
    for _ in range(num_iterations):
        x_next = np.dot(W, x)
        norm = np.linalg.norm(x_next)
        if norm > 1e-12:
            x = x_next / norm
        else:
            break

    # Normalize scores to [0, 1]
    max_score = np.max(x)
    if max_score > 0:
        scores = x / max_score
    else:
        scores = x

    # Greedy discretization enforcing strict 1-to-1 matching
    sorted_order = np.argsort(scores)[::-1]
    used_a = set()
    used_b = set()
    selected_matches = []

    for idx in sorted_order:
        score = float(scores[idx])
        if score < ambiguity_threshold:
            continue

        i, j, fid = candidates[idx]
        if i not in used_a and j not in used_b:
            selected_matches.append((i, j, score, fid))
            used_a.add(i)
            used_b.add(j)

    return selected_matches


class PairwiseEntropyMatcher:
    """
    High-level engine for matching two images using pairwise Von Neumann entropy.
    """

    def __init__(
        self,
        matching_config: Optional[MatchingConfig] = None,
        geometric_config: Optional[GeometricConfig] = None,
    ):
        self.matching_cfg = matching_config or MatchingConfig()
        self.geom_cfg = geometric_config or GeometricConfig()

    def match(
        self,
        rep_a: ImageEntropyRepresentation,
        rep_b: ImageEntropyRepresentation,
    ) -> MatchResult:
        """
        Execute full matching pipeline between Image A and Image B.

        Pipeline:
        1. Extract candidate seed correspondences via quantum fidelity.
        2. Construct relational compatibility graph using pairwise entropy differences.
        3. Solve spectral assignment to reject incoherent and ambiguous matches.
        4. Execute RANSAC geometric verification.
        5. Compute calibrated image-level similarity score.

        Args:
            rep_a: Pairwise entropy representation of Image A.
            rep_b: Pairwise entropy representation of Image B.

        Returns:
            MatchResult object.
        """
        n_a = rep_a.num_keypoints
        n_b = rep_b.num_keypoints

        # Handle empty images
        if n_a == 0 or n_b == 0:
            return MatchResult(
                matches=[],
                pts_a=np.empty((0, 2), dtype=np.float64),
                pts_b=np.empty((0, 2), dtype=np.float64),
                inlier_mask=np.empty(0, dtype=bool),
                inlier_count=0,
                similarity_score=0.0,
                mean_entropy_error=float("inf"),
                geometric_status="empty_keypoints",
                model_matrix=None,
                num_kps_a=n_a,
                num_kps_b=n_b,
            )

        # 1. Candidate Seeds
        candidates = generate_candidate_seeds(
            rep_a.quantum_states,
            rep_b.quantum_states,
            seed_method=self.matching_cfg.seed_method,
            ratio_threshold=self.matching_cfg.ratio_threshold,
            top_k=self.matching_cfg.top_k_candidates,
        )

        if len(candidates) == 0:
            return MatchResult(
                matches=[],
                pts_a=np.empty((0, 2), dtype=np.float64),
                pts_b=np.empty((0, 2), dtype=np.float64),
                inlier_mask=np.empty(0, dtype=bool),
                inlier_count=0,
                similarity_score=0.0,
                mean_entropy_error=float("inf"),
                geometric_status="no_seed_matches",
                model_matrix=None,
                num_kps_a=n_a,
                num_kps_b=n_b,
            )

        # 2. Relational Compatibility Matrix W
        W = build_compatibility_matrix(
            candidates,
            rep_a.entropy_matrix,
            rep_b.entropy_matrix,
            entropy_sigma=self.matching_cfg.entropy_sigma,
            relational_weight=self.matching_cfg.relational_weight,
        )

        # 3. Spectral Assignment & Conflict Resolution
        selected = solve_spectral_assignment(
            W,
            candidates,
            num_iterations=self.matching_cfg.power_iterations,
            ambiguity_threshold=self.matching_cfg.ambiguity_threshold,
        )

        if len(selected) == 0:
            return MatchResult(
                matches=[],
                pts_a=np.empty((0, 2), dtype=np.float64),
                pts_b=np.empty((0, 2), dtype=np.float64),
                inlier_mask=np.empty(0, dtype=bool),
                inlier_count=0,
                similarity_score=0.0,
                mean_entropy_error=float("inf"),
                geometric_status="rejected_by_spectral_matching",
                model_matrix=None,
                num_kps_a=n_a,
                num_kps_b=n_b,
            )

        # Build coordinate arrays
        pts_a = np.array([rep_a.descriptors[i].pt for i, _, _, _ in selected], dtype=np.float64)
        pts_b = np.array([rep_b.descriptors[j].pt for _, j, _, _ in selected], dtype=np.float64)

        # Compute mean relational entropy difference between matched pairs
        entropy_errors = []
        for u in range(len(selected)):
            i_u, j_u, _, _ = selected[u]
            for v in range(u + 1, len(selected)):
                i_v, j_v, _, _ = selected[v]
                err = abs(float(rep_a.entropy_matrix[i_u, i_v]) - float(rep_b.entropy_matrix[j_u, j_v]))
                entropy_errors.append(err)
        mean_err = float(np.mean(entropy_errors)) if len(entropy_errors) > 0 else 0.0

        # 4. Geometric Verification via RANSAC
        geom_result = verify_matches_ransac(
            pts_a,
            pts_b,
            model=self.geom_cfg.model,
            threshold=self.geom_cfg.ransac_threshold,
            max_iters=self.geom_cfg.max_iterations,
            confidence=self.geom_cfg.confidence,
            min_inliers=self.geom_cfg.min_inliers,
        )

        inlier_mask = geom_result["inlier_mask"]
        inlier_count = geom_result["inlier_count"]
        model_mat = geom_result["model_matrix"]

        # 5. Image-Level Similarity Score Formulation
        # Combines relational entropy agreement, inlier ratio, and match coverage
        min_kps = min(n_a, n_b)
        coverage_ratio = float(len(selected) / min_kps) if min_kps > 0 else 0.0
        inlier_ratio = float(inlier_count / len(selected)) if len(selected) > 0 else 0.0

        relational_quality = float(np.exp(-mean_err / self.matching_cfg.entropy_sigma))

        if inlier_count >= self.geom_cfg.min_inliers:
            similarity = (
                0.40 * relational_quality +
                0.35 * inlier_ratio +
                0.25 * min(1.0, coverage_ratio)
            ) * 100.0
        else:
            # Penalize geometrically unverified matches
            similarity = (
                0.60 * relational_quality +
                0.40 * min(1.0, coverage_ratio)
            ) * 50.0  # Cap at 50% if unverified

        similarity_clamped = float(np.clip(similarity, 0.0, 100.0))

        return MatchResult(
            matches=selected,
            pts_a=pts_a,
            pts_b=pts_b,
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            similarity_score=similarity_clamped,
            mean_entropy_error=mean_err,
            geometric_status=geom_result["status"],
            model_matrix=model_mat,
            num_kps_a=n_a,
            num_kps_b=n_b,
        )
