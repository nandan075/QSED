"""
Unit tests for QCSI Multimodal Fusion and Calibration Engine.
"""

import numpy as np
import pytest
from qcsi.fusion import (
    extract_feature_vector,
    CalibratedFusionModel,
    explain_similarity,
    FEATURE_NAMES,
)


def create_mock_branch_results():
    local_res = {
        "rotation": "0 deg",
        "geometry": {
            "inlier_count": 18,
            "inlier_ratio": 0.22,
            "mean_inlier_fidelity": 0.88,
            "mean_trace_distance": 0.34,
        },
        "uhlmann_fidelity": 0.75,
        "qjsd": 0.12,
        "hs_cosine": 0.82,
    }
    global_res = {
        "classical_cosine": 0.85,
        "quantum_fidelity": 0.78,
    }
    return local_res, global_res


def test_extract_feature_vector():
    loc, glob = create_mock_branch_results()
    feat = extract_feature_vector(loc, glob)
    assert feat.shape == (9,)
    assert len(FEATURE_NAMES) == 9
    assert feat[0] == 18.0  # inlier_count
    assert feat[7] == 0.85  # dino_cos
    assert feat[8] == 0.78  # q_global_fid


def test_calibrated_fusion_model():
    model = CalibratedFusionModel(use_isotonic=True)
    loc, glob = create_mock_branch_results()
    feat = extract_feature_vector(loc, glob)

    # Test untrained heuristic prediction
    prob_unfitted = model.predict_probability(feat)
    assert 0.0 <= prob_unfitted <= 1.0
    assert prob_unfitted > 0.6  # High match features

    # Train model on synthetic data
    X = np.array([
        [20, 0.25, 0.9, 0.3, 0.8, 0.1, 0.85, 0.9, 0.8],
        [15, 0.20, 0.85, 0.35, 0.75, 0.15, 0.80, 0.85, 0.75],
        [22, 0.28, 0.92, 0.28, 0.82, 0.08, 0.88, 0.92, 0.82],
        [0, 0.0, 0.0, 1.0, 0.1, 0.8, 0.2, 0.1, 0.05],
        [1, 0.02, 0.3, 0.9, 0.15, 0.75, 0.25, 0.2, 0.1],
        [0, 0.0, 0.0, 1.0, 0.05, 0.85, 0.15, 0.05, 0.02],
    ], dtype=np.float64)
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)

    model.fit(X, y)
    assert model.is_fitted

    prob_pos = model.predict_probability(X[0])
    prob_neg = model.predict_probability(X[3])
    assert prob_pos > prob_neg


def test_explain_similarity():
    loc, glob = create_mock_branch_results()
    explanation = explain_similarity(loc, glob, calibrated_prob=0.88)

    assert explanation["calibrated_similarity_pct"] == 88.0
    assert explanation["is_match"] is True
    assert "Strong Match" in explanation["verdict"]
    assert "components" in explanation
    assert "evidence" in explanation
