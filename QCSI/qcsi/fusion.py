"""
QCSI Multimodal Fusion and Calibration Engine.

Aggregates Branch A (Local Geometry) and Branch B (Global Semantics) into a
calibrated similarity percentage [0%, 100%] with comprehensive explainability.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


FEATURE_NAMES = [
    "inlier_count",
    "inlier_ratio",
    "mean_inlier_fidelity",
    "mean_trace_distance",
    "uhlmann_fidelity",
    "qjsd",
    "hs_cosine",
    "dinov2_cosine",
    "quantum_global_fidelity",
]


def extract_feature_vector(
    local_result: Dict[str, Any],
    global_result: Dict[str, Any]
) -> np.ndarray:
    """
    Constructs the 9-dimensional multimodal feature vector from Branch A and Branch B outputs.
    """
    geo = local_result.get("geometry", {})
    inlier_count = float(geo.get("inlier_count", 0))
    inlier_ratio = float(geo.get("inlier_ratio", 0.0))
    mean_inlier_fid = float(geo.get("mean_inlier_fidelity", 0.0))
    mean_trace_dist = float(geo.get("mean_trace_distance", 1.0))

    uhlmann_fid = float(local_result.get("uhlmann_fidelity", 0.0))
    qjsd = float(local_result.get("qjsd", 1.0))
    hs_cos = float(local_result.get("hs_cosine", 0.0))

    dino_cos = float(global_result.get("classical_cosine", 0.0))
    q_global_fid = float(global_result.get("quantum_fidelity", 0.0))

    features = np.array([
        inlier_count,
        inlier_ratio,
        mean_inlier_fid,
        mean_trace_dist,
        uhlmann_fid,
        qjsd,
        hs_cos,
        dino_cos,
        q_global_fid,
    ], dtype=np.float64)

    return features


class CalibratedFusionModel:
    """
    Calibrated fusion classifier using Logistic Regression + Isotonic Calibration.
    Produces well-calibrated probabilities P(same | features) in [0.0, 1.0].
    """

    def __init__(self, use_isotonic: bool = True):
        self.use_isotonic = use_isotonic
        self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits base classifier and isotonic calibrator on training pairs.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)

        if len(np.unique(y)) < 2:
            # Degenerate case (only one class)
            self.is_fitted = False
            return

        self.classifier.fit(X, y)
        probs = self.classifier.predict_proba(X)[:, 1]

        if self.use_isotonic and len(np.unique(probs)) > 2:
            self.calibrator.fit(probs, y)

        self.is_fitted = True

    def predict_probability(self, feature_vector: np.ndarray) -> float:
        """
        Predicts calibrated match probability in [0.0, 1.0].
        If not fitted, uses robust heuristic fusion baseline.
        """
        feat = np.asarray(feature_vector, dtype=np.float64)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)

        if not self.is_fitted:
            return self._heuristic_fallback_probability(feat[0])

        raw_prob = self.classifier.predict_proba(feat)[:, 1][0]
        if self.use_isotonic and hasattr(self.calibrator, "predict"):
            calibrated = float(self.calibrator.predict([raw_prob])[0])
        else:
            calibrated = float(raw_prob)

        return float(np.clip(calibrated, 0.0, 1.0))

    def _heuristic_fallback_probability(self, f: np.ndarray) -> float:
        """
        Physically motivated prior when no training labels are provided.
        Combines:
        - Geometry (inlier count & ratio)
        - Semantics (DINOv2 & quantum global fidelity)
        - Density operator metrics (Uhlmann fidelity, QJSD)
        """
        inlier_count, inlier_ratio, mean_inlier_fid, mean_trace_dist = f[0], f[1], f[2], f[3]
        uhlmann_fid, qjsd, hs_cos = f[4], f[5], f[6]
        dino_cos, q_global_fid = f[7], f[8]

        # Geometry score: saturates around 20 inliers or 25% inlier ratio
        geom_score = np.clip(inlier_count / 15.0, 0.0, 1.0) * 0.6 + np.clip(inlier_ratio / 0.2, 0.0, 1.0) * 0.4

        # Semantics score
        dino_clamped = max(0.0, (dino_cos + 0.2) / 1.2)  # DINOv2 cosine typically ranges from ~0.2 to 1.0 for same scene
        sem_score = 0.5 * dino_clamped + 0.5 * q_global_fid

        # Quantum info score
        q_info_score = 0.4 * uhlmann_fid + 0.4 * (1.0 - qjsd) + 0.2 * hs_cos

        # Weighted combination: 45% geometry, 35% semantics, 20% quantum density
        combined = 0.45 * geom_score + 0.35 * sem_score + 0.20 * q_info_score

        # Sigmoid shape around decision boundary
        prob = 1.0 / (1.0 + np.exp(-10.0 * (combined - 0.45)))
        return float(np.clip(prob, 0.0, 1.0))

    def save(self, filepath: str):
        """Saves fitted calibration models to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "calibrator": self.calibrator,
                "is_fitted": self.is_fitted,
                "use_isotonic": self.use_isotonic,
            }, f)

    def load(self, filepath: str):
        """Loads fitted calibration models from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.classifier = data["classifier"]
            self.calibrator = data["calibrator"]
            self.is_fitted = data["is_fitted"]
            self.use_isotonic = data.get("use_isotonic", True)


def explain_similarity(
    local_result: Dict[str, Any],
    global_result: Dict[str, Any],
    calibrated_prob: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Builds human-readable explanation and component decomposition.
    """
    geo = local_result.get("geometry", {})
    inliers = int(geo.get("inlier_count", 0))
    inlier_ratio = float(geo.get("inlier_ratio", 0.0))
    mean_fid = float(geo.get("mean_inlier_fidelity", 0.0))

    uhlmann_fid = float(local_result.get("uhlmann_fidelity", 0.0))
    qjsd = float(local_result.get("qjsd", 1.0))

    dino_cos = float(global_result.get("classical_cosine", 0.0))
    q_global_fid = float(global_result.get("quantum_fidelity", 0.0))

    pct = round(calibrated_prob * 100.0, 2)

    # Component sub-scores in [0, 100]
    geom_component = min(100.0, (inliers / 15.0) * 60.0 + (inlier_ratio / 0.25) * 40.0)
    sem_component = max(0.0, min(100.0, (dino_cos * 50.0 + q_global_fid * 50.0)))
    quantum_component = max(0.0, min(100.0, (uhlmann_fid * 50.0 + (1.0 - qjsd) * 50.0)))

    if calibrated_prob >= 0.75:
        verdict = "Strong Match (High Confidence Same Scene / Object)"
        reasoning = (
            f"The image pair exhibits strong geometric alignment with {inliers} verified RANSAC inliers "
            f"(mean quantum fidelity {mean_fid*100:.1f}%) and high global semantic similarity "
            f"(DINOv2 cosine: {dino_cos:.3f}, ZZFeatureMap fidelity: {q_global_fid*100:.1f}%)."
        )
    elif calibrated_prob >= threshold:
        verdict = "Moderate Match (Likely Same Scene / Transformed)"
        reasoning = (
            f"The image pair displays consistent visual features with {inliers} inliers, "
            f"though partial viewpoint, scale, or lighting transformations are present."
        )
    else:
        verdict = "Non-Match (Different Scenes / Objects)"
        reasoning = (
            f"Insufficient geometric consistency ({inliers} inliers) and low semantic overlap "
            f"(DINOv2 cosine: {dino_cos:.3f}, QJSD: {qjsd:.4f})."
        )

    return {
        "calibrated_similarity_pct": pct,
        "calibrated_probability": calibrated_prob,
        "is_match": bool(calibrated_prob >= threshold),
        "verdict": verdict,
        "reasoning": reasoning,
        "components": {
            "geometric_score": round(geom_component, 2),
            "semantic_score": round(sem_component, 2),
            "quantum_state_score": round(quantum_component, 2),
        },
        "evidence": {
            "inlier_count": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "mean_inlier_fidelity": round(mean_fid, 4),
            "uhlmann_fidelity": round(uhlmann_fid, 4),
            "qjsd": round(qjsd, 4),
            "dinov2_cosine": round(dino_cos, 4),
            "quantum_global_fidelity": round(q_global_fid, 4),
            "rotation_detected": local_result.get("rotation", "0 deg"),
        }
    }
