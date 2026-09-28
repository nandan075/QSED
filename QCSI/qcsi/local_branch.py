"""
QCSI Branch A: Local Geometric Matching via Quantum Fidelity & RANSAC.

Pipeline:
1. SIFT -> RootSIFT -> Centering.
2. 7-Qubit Amplitude Encoding (|psi_i> in C^128).
3. Quantum Fidelity Kernel K = |<psi|phi>|^2 & Trace Distance D = sqrt(1 - K).
4. Mutual-NN + Lowe's Ratio Test.
5. Geometric Verification via RANSAC (Homography / Fundamental Matrix).
6. Mixed-State Metrics: Ensemble Density Matrices, HS-Cosine, Uhlmann Fidelity, QJSD.
7. Optional 4-Direction Cardinal Rotation Search.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from qcsi.quantum import (
    amplitude_encode_pure_states,
    compute_pairwise_quantum_metrics,
    build_ensemble_density_matrix,
    von_neumann_entropy,
    quantum_jensen_shannon_divergence,
    uhlmann_jozsa_fidelity,
    hilbert_schmidt_cosine,
)


class LocalGeometryMatcher:
    """Local keypoint extraction, quantum state mapping, and geometric verification."""

    def __init__(
        self,
        max_keypoints: int = 500,
        ratio_threshold: float = 0.8,
        ransac_threshold: float = 3.0,
        verification_model: str = "homography",  # 'homography' or 'fundamental'
        center_descriptors: bool = True,
        search_4directions: bool = False,
    ):
        self.max_keypoints = max_keypoints
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.verification_model = verification_model.lower()
        self.center_descriptors = center_descriptors
        self.search_4directions = search_4directions

        try:
            self.detector = cv2.SIFT_create(nfeatures=self.max_keypoints)
            self._detector_type = "SIFT"
        except Exception:
            self.detector = cv2.ORB_create(nfeatures=self.max_keypoints)
            self._detector_type = "ORB"

    def extract_features(self, gray_img: np.ndarray) -> Dict[str, Any]:
        """
        Extracts keypoints and converts descriptors to RootSIFT pure quantum states.
        
        Returns dictionary containing:
        - keypoints: list of cv2.KeyPoint
        - raw_descriptors: np.ndarray
        - psi: (N, 128) amplitude-encoded quantum pure states
        - responses: keypoint response scores
        - density_matrix: (128, 128) ensemble density operator rho
        - entropy: Von Neumann entropy S(rho)
        """
        kps, descs = self.detector.detectAndCompute(gray_img, None)

        dim = 128
        if descs is None or len(descs) == 0:
            rho = build_ensemble_density_matrix(None, dim=dim)
            return {
                "keypoints": [],
                "raw_descriptors": np.empty((0, dim)),
                "psi": np.empty((0, dim)),
                "responses": np.empty((0,)),
                "pts": np.empty((0, 2)),
                "density_matrix": rho,
                "entropy": von_neumann_entropy(rho),
                "num_kps": 0,
            }

        # Pad ORB descriptors if needed
        if descs.shape[1] != dim:
            padded = np.zeros((descs.shape[0], dim), dtype=np.float32)
            padded[:, :min(dim, descs.shape[1])] = descs[:, :min(dim, descs.shape[1])]
            descs = padded

        # 1. RootSIFT Transformation: L1 norm -> sqrt
        l1_norm = np.linalg.norm(descs, ord=1, axis=1, keepdims=True)
        l1_norm = np.maximum(l1_norm, 1e-12)
        root_sift = np.sqrt(descs / l1_norm)

        # 2. Centering (zero-mean per descriptor)
        if self.center_descriptors:
            mean = np.mean(root_sift, axis=1, keepdims=True)
            root_sift = root_sift - mean

        # 3. 7-Qubit Amplitude Encoding: L2 normalization to unit state |psi>
        psi = amplitude_encode_pure_states(root_sift)

        # Keypoint weights for mixed state ensemble
        responses = np.array([kp.response for kp in kps], dtype=np.float64)
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)

        # Density operator rho = sum_i w_i |psi_i><psi_i|
        rho = build_ensemble_density_matrix(psi, weights=responses, dim=dim)
        entropy = von_neumann_entropy(rho)

        return {
            "keypoints": kps,
            "raw_descriptors": descs,
            "psi": psi,
            "responses": responses,
            "pts": pts,
            "density_matrix": rho,
            "entropy": entropy,
            "num_kps": len(kps),
        }

    def match_pure_states(
        self,
        feat_A: Dict[str, Any],
        feat_B: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Performs mutual-NN matching and Lowe's ratio test on quantum trace distances.
        """
        psi_A = feat_A["psi"]
        psi_B = feat_B["psi"]

        if len(psi_A) == 0 or len(psi_B) == 0:
            return {
                "matches": [],
                "pts_A": np.empty((0, 2)),
                "pts_B": np.empty((0, 2)),
                "fidelities": np.empty((0,)),
                "trace_distances": np.empty((0,)),
            }

        # Vectorized quantum kernels
        fid_matrix, dist_matrix = compute_pairwise_quantum_metrics(psi_A, psi_B)
        M, N = dist_matrix.shape

        matches = []
        fidelities = []
        trace_dists = []

        # Forward matching: A -> B
        for i in range(M):
            dists_row = dist_matrix[i]
            if N < 2:
                best_j = int(np.argmin(dists_row))
                matches.append((i, best_j))
                fidelities.append(fid_matrix[i, best_j])
                trace_dists.append(dists_row[best_j])
                continue

            # Two smallest trace distances
            sorted_indices = np.argsort(dists_row)[:2]
            j1, j2 = sorted_indices[0], sorted_indices[1]
            d1, d2 = dists_row[j1], dists_row[j2]

            # Ratio test on trace distance
            if d2 > 0 and (d1 / d2) <= self.ratio_threshold:
                # Mutual Nearest Neighbor check: is i also best for j1?
                best_i_for_j1 = int(np.argmin(dist_matrix[:, j1]))
                if best_i_for_j1 == i:
                    matches.append((i, j1))
                    fidelities.append(fid_matrix[i, j1])
                    trace_dists.append(d1)

        matched_pts_A = np.array([feat_A["pts"][m[0]] for m in matches], dtype=np.float32) if matches else np.empty((0, 2))
        matched_pts_B = np.array([feat_B["pts"][m[1]] for m in matches], dtype=np.float32) if matches else np.empty((0, 2))

        return {
            "matches": matches,
            "pts_A": matched_pts_A,
            "pts_B": matched_pts_B,
            "fidelities": np.array(fidelities, dtype=np.float64),
            "trace_distances": np.array(trace_dists, dtype=np.float64),
        }

    def verify_geometry(
        self,
        pts_A: np.ndarray,
        pts_B: np.ndarray,
        fidelities: np.ndarray,
        num_kps_A: int,
        num_kps_B: int
    ) -> Dict[str, Any]:
        """
        RANSAC geometric verification (Homography or Fundamental matrix).
        """
        min_pts = 4 if self.verification_model == "homography" else 8
        n_matches = len(pts_A)

        if n_matches < min_pts:
            return {
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "match_inlier_ratio": 0.0,
                "mean_inlier_fidelity": 0.0,
                "mean_trace_distance": 1.0,
                "model_matrix": None,
                "inlier_mask": np.zeros(n_matches, dtype=bool),
                "inlier_pts_A": np.empty((0, 2)),
                "inlier_pts_B": np.empty((0, 2)),
            }

        if self.verification_model == "homography":
            matrix, mask = cv2.findHomography(
                pts_A, pts_B, cv2.RANSAC, self.ransac_threshold
            )
        else:
            matrix, mask = cv2.findFundamentalMat(
                pts_A, pts_B, cv2.FM_RANSAC, self.ransac_threshold
            )

        if mask is None:
            mask = np.zeros(n_matches, dtype=np.uint8)

        inlier_mask = (mask.ravel() == 1)
        inlier_count = int(np.sum(inlier_mask))

        min_kps = max(min(num_kps_A, num_kps_B), 1)
        inlier_ratio = float(inlier_count / min_kps)
        match_inlier_ratio = float(inlier_count / max(n_matches, 1))

        if inlier_count > 0:
            mean_inlier_fid = float(np.mean(fidelities[inlier_mask]))
            mean_trace_dist = float(np.mean(np.sqrt(np.maximum(0.0, 1.0 - fidelities[inlier_mask]))))
        else:
            mean_inlier_fid = 0.0
            mean_trace_dist = 1.0

        return {
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "match_inlier_ratio": match_inlier_ratio,
            "mean_inlier_fidelity": mean_inlier_fid,
            "mean_trace_distance": mean_trace_dist,
            "model_matrix": matrix,
            "inlier_mask": inlier_mask,
            "inlier_pts_A": pts_A[inlier_mask] if inlier_count > 0 else np.empty((0, 2)),
            "inlier_pts_B": pts_B[inlier_mask] if inlier_count > 0 else np.empty((0, 2)),
        }

    def compute_local_similarity(
        self,
        gray_A: np.ndarray,
        gray_B: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes full Branch A pipeline on preprocessed grayscale image pair.
        """
        feat_A = self.extract_features(gray_A)

        rotations = [("0 deg", None)]
        if self.search_4directions:
            rotations.extend([
                ("90 deg CW", cv2.ROTATE_90_CLOCKWISE),
                ("180 deg", cv2.ROTATE_180),
                ("270 deg CCW", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ])

        best_result = None
        best_score = -1.0

        for rot_label, rot_code in rotations:
            rot_B = gray_B.copy() if rot_code is None else cv2.rotate(gray_B, rot_code)
            feat_B = self.extract_features(rot_B)

            # Mixed-state metrics
            rho_A, rho_B = feat_A["density_matrix"], feat_B["density_matrix"]
            uhlmann_fid = uhlmann_jozsa_fidelity(rho_A, rho_B)
            qjsd = quantum_jensen_shannon_divergence(rho_A, rho_B)
            hs_cos = hilbert_schmidt_cosine(rho_A, rho_B)

            # Keypoint pure state matching
            matched = self.match_pure_states(feat_A, feat_B)
            geo = self.verify_geometry(
                matched["pts_A"],
                matched["pts_B"],
                matched["fidelities"],
                feat_A["num_kps"],
                feat_B["num_kps"],
            )

            # Heuristic selection score for cardinal rotation search
            selection_score = geo["inlier_count"] * 2.0 + uhlmann_fid + (1.0 - qjsd)

            res = {
                "rotation": rot_label,
                "feat_A": feat_A,
                "feat_B": feat_B,
                "matched": matched,
                "geometry": geo,
                "uhlmann_fidelity": uhlmann_fid,
                "qjsd": qjsd,
                "hs_cosine": hs_cos,
                "entropy_A": feat_A["entropy"],
                "entropy_B": feat_B["entropy"],
            }

            if selection_score > best_score:
                best_score = selection_score
                best_result = res

        return best_result
