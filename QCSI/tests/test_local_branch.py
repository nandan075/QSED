"""
Unit tests for QCSI Branch A (Local Geometry Matcher).
"""

import numpy as np
import cv2
import pytest
from qcsi.local_branch import LocalGeometryMatcher


def create_synthetic_test_pair():
    img_A = np.zeros((300, 300), dtype=np.uint8) + 220
    cv2.circle(img_A, (80, 80), 30, 40, -1)
    cv2.rectangle(img_A, (150, 60), (230, 140), 60, -1)
    cv2.putText(img_A, "QCSI", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    # Perspective transformed version
    pts1 = np.float32([[20, 20], [280, 20], [280, 280], [20, 280]])
    pts2 = np.float32([[30, 40], [260, 10], [290, 260], [10, 270]])
    H = cv2.getPerspectiveTransform(pts1, pts2)
    img_B = cv2.warpPerspective(img_A, H, (300, 300), borderValue=220)
    return img_A, img_B


def test_local_branch_extraction():
    matcher = LocalGeometryMatcher(max_keypoints=200)
    img_A, _ = create_synthetic_test_pair()

    feat = matcher.extract_features(img_A)
    assert feat["num_kps"] > 0
    assert feat["psi"].shape[1] == 128
    assert feat["density_matrix"].shape == (128, 128)
    assert feat["entropy"] >= 0.0


def test_local_branch_matching_and_ransac():
    matcher = LocalGeometryMatcher(max_keypoints=300, ratio_threshold=0.85)
    img_A, img_B = create_synthetic_test_pair()

    res = matcher.compute_local_similarity(img_A, img_B)
    geo = res["geometry"]
    matched = res["matched"]

    assert len(matched["matches"]) > 0
    assert geo["inlier_count"] >= 4
    assert geo["inlier_ratio"] > 0.0
    assert geo["mean_inlier_fidelity"] > 0.5
    assert res["uhlmann_fidelity"] > 0.0
    assert 0.0 <= res["qjsd"] <= 1.0
