import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature.directional_variation import angular_difference
from src.feature.neighborhood import extract_neighborhood
from src.feature.density_matrix import build_intensity_density_matrix, build_covariance_density_matrix, verify_density_matrix
from src.feature.von_neumann import von_neumann_entropy, compute_entropy_map
from src.feature.keypoint_detection import Keypoint
from src.feature.ranking import rank_keypoints

def test_angular_difference_same():
    assert angular_difference(45.0, 45.0) == 0.0

def test_angular_difference_opposite():
    assert angular_difference(0.0, 90.0) == 90.0

def test_angular_difference_wraparound():
    assert angular_difference(0.0, 157.5) == 22.5

def test_directional_variation_uniform():
    # uniform directions
    dom_dir = np.zeros((5, 5))
    grad_mag = np.ones((5, 5))
    from src.feature.directional_variation import compute_directional_variation
    var = compute_directional_variation(dom_dir, grad_mag)
    assert np.allclose(var, 0.0)

def test_directional_variation_varied():
    # Use direction INDICES (0..7), not angles. Index 0→0°, 2→45°, 4→90°, 6→135°, 7→157.5°
    dom_dir = np.array([[0, 2, 4], [6, 0, 1], [2, 4, 6]], dtype=np.int64)
    grad_mag = np.ones((3, 3))
    from src.feature.directional_variation import compute_directional_variation
    var = compute_directional_variation(dom_dir, grad_mag, neighborhood_size=3)
    assert var[1, 1] > 0.0

def test_extract_neighborhood():
    img = np.arange(25).reshape(5, 5)
    neigh = extract_neighborhood(img, 2, 2, size=3)
    expected = np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]])
    assert np.array_equal(neigh, expected)

def test_intensity_density_matrix_properties():
    neigh = np.ones((3, 3))
    rho = build_intensity_density_matrix(neigh)
    assert verify_density_matrix(rho)

def test_intensity_density_matrix_uniform():
    neigh = np.ones((3, 3))
    rho = build_intensity_density_matrix(neigh)
    S = von_neumann_entropy(rho)
    assert np.isclose(S, np.log2(9))

def test_pure_state_would_give_zero_entropy():
    # pure state |0><0|
    rho = np.zeros((9, 9))
    rho[0, 0] = 1.0
    S = von_neumann_entropy(rho)
    assert np.isclose(S, 0.0, atol=1e-5)

def test_von_neumann_entropy_uniform():
    rho = np.eye(9) / 9
    S = von_neumann_entropy(rho)
    assert np.isclose(S, np.log2(9))

def test_von_neumann_entropy_deterministic():
    rho = np.zeros((9, 9))
    rho[0, 0] = 1.0
    S = von_neumann_entropy(rho)
    assert np.isclose(S, 0.0, atol=1e-5)

def test_entropy_map_shape():
    img = np.zeros((10, 10))
    emap, enmap = compute_entropy_map(img)
    assert emap.shape == (10, 10)
    assert enmap.shape == (10, 10)

def test_covariance_density_matrix():
    features = np.random.rand(8, 9)
    rho = build_covariance_density_matrix(features)
    assert verify_density_matrix(rho)

def test_keypoint_detection_synthetic():
    grad_mag = np.zeros((16, 16))
    dom_dir = np.zeros((16, 16))
    edge_mask = np.zeros((16, 16))
    dir_var = np.zeros((16, 16))
    kp_score = np.zeros((16, 16))
    
    grad_mag[5, 5] = 1.0
    edge_mask[5, 5] = 1.0
    dir_var[5, 5] = 1.0
    kp_score[5, 5] = 1.0
    
    from src.feature.keypoint_detection import detect_keypoints
    kps = detect_keypoints(grad_mag, dom_dir, edge_mask, dir_var, kp_score, grad_threshold_ratio=0.0)
    assert len(kps) > 0
    assert kps[0].x == 5 and kps[0].y == 5

def test_ranking():
    kps = [Keypoint(0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5), Keypoint(1, 1, 0, 0, 0, 0.8, 0.8, 0.8, 0.8)]
    ranked = rank_keypoints(kps)
    assert ranked[0].x == 1

def test_verify_density_matrix():
    rho = np.eye(9) / 9
    assert verify_density_matrix(rho)
    rho_invalid = np.eye(9)
    assert not verify_density_matrix(rho_invalid)
