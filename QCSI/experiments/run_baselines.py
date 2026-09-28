"""
QCSI Baseline Evaluation Suite (A0 - A3).

Implements:
- A0: Perceptual Hash (pHash) similarity.
- A1: Fully Classical SIFT + RANSAC.
- A2: Frozen DINOv2 Cosine Similarity.
- A3: Legacy QSED score (0.6 * Fidelity + 0.4 * (1 - sqrt(QJSD))).
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2

from qcsi.preprocess import ImagePreprocessor
from qcsi.global_branch import DINOv2Extractor
from qcsi.quantum import (
    amplitude_encode_pure_states,
    build_ensemble_density_matrix,
    uhlmann_jozsa_fidelity,
    quantum_jensen_shannon_divergence,
)


class BaselineEvaluator:
    """Computes scores for standard classical and legacy baseline systems."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.dino_extractor = DINOv2Extractor()
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
        except Exception:
            self.sift = cv2.ORB_create(nfeatures=500)

    def evaluate_A0_phash(self, img_A: np.ndarray, img_B: np.ndarray) -> float:
        """
        A0: Perceptual Hash (DCT-based pHash).
        Normalized similarity = 1.0 - (Hamming_distance / 64.0).
        """
        def _compute_phash(img: np.ndarray) -> np.ndarray:
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            dct = cv2.dct(small)
            dct_low = dct[:8, :8]
            median = np.median(dct_low[1:])
            hash_bits = (dct_low > median).flatten()
            return hash_bits

        hA = _compute_phash(img_A)
        hB = _compute_phash(img_B)
        hamming_dist = np.count_nonzero(hA != hB)
        sim = 1.0 - (hamming_dist / 64.0)
        return float(np.clip(sim, 0.0, 1.0))

    def evaluate_A1_classical_sift(self, gray_A: np.ndarray, gray_B: np.ndarray) -> float:
        """
        A1: SIFT + Classical FLANN/BFMatcher + Lowe's Ratio Test + RANSAC Homography.
        Score = inliers / min(kps_A, kps_B).
        """
        kp1, des1 = self.sift.detectAndCompute(gray_A, None)
        kp2, des2 = self.sift.detectAndCompute(gray_B, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2 and m_pair[0].distance < 0.8 * m_pair[1].distance:
                good_matches.append(m_pair[0])

        if len(good_matches) < 4:
            return 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if mask is None:
            return 0.0

        inliers = int(np.sum(mask))
        min_kps = max(min(len(kp1), len(kp2)), 1)
        score = float(np.clip(inliers / float(min_kps), 0.0, 1.0))
        return score

    def evaluate_A2_dinov2_cosine(self, img_A: np.ndarray, img_B: np.ndarray) -> float:
        """
        A2: Frozen DINOv2 ViT-B/14 768-d CLS token cosine similarity.
        Normalized to [0.0, 1.0].
        """
        eA = self.dino_extractor.extract_embedding(img_A)
        eB = self.dino_extractor.extract_embedding(img_B)
        cos_sim = float(np.dot(eA, eB))
        # Rescale from typical range [-0.2, 1.0] to [0.0, 1.0]
        norm_score = (cos_sim + 0.2) / 1.2
        return float(np.clip(norm_score, 0.0, 1.0))

    def evaluate_A3_legacy_qsed(self, gray_A: np.ndarray, gray_B: np.ndarray) -> float:
        """
        A3: Old QSED score from quantum_sift_matcher.py:
        score = 0.6 * Fidelity + 0.4 * (1 - sqrt(QJSD)).
        """
        kp1, des1 = self.sift.detectAndCompute(gray_A, None)
        kp2, des2 = self.sift.detectAndCompute(gray_B, None)

        psi1 = amplitude_encode_pure_states(des1) if des1 is not None else np.empty((0, 128))
        psi2 = amplitude_encode_pure_states(des2) if des2 is not None else np.empty((0, 128))

        w1 = np.array([k.response for k in kp1]) if kp1 else None
        w2 = np.array([k.response for k in kp2]) if kp2 else None

        rho_A = build_ensemble_density_matrix(psi1, weights=w1, dim=128)
        rho_B = build_ensemble_density_matrix(psi2, weights=w2, dim=128)

        fid = uhlmann_jozsa_fidelity(rho_A, rho_B)
        qjsd = quantum_jensen_shannon_divergence(rho_A, rho_B)
        sim_qjsd = max(0.0, 1.0 - np.sqrt(qjsd))

        qsed_score = 0.6 * fid + 0.4 * sim_qjsd
        return float(np.clip(qsed_score, 0.0, 1.0))

    def evaluate_all_baselines(self, img_A: np.ndarray, img_B: np.ndarray) -> Dict[str, float]:
        """Runs all 4 baselines on an image pair."""
        prep = self.preprocessor.preprocess_pair(img_A, img_B)
        return {
            "A0_phash": self.evaluate_A0_phash(prep["rgb_A"], prep["rgb_B"]),
            "A1_classical_sift": self.evaluate_A1_classical_sift(prep["gray_A"], prep["gray_B"]),
            "A2_dinov2_cosine": self.evaluate_A2_dinov2_cosine(prep["rgb_A"], prep["rgb_B"]),
            "A3_legacy_qsed": self.evaluate_A3_legacy_qsed(prep["gray_A"], prep["gray_B"]),
        }
