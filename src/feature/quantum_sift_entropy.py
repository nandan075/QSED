"""
Quantum SIFT/SURF & Von Neumann Entropy Feature Matching Module.
Integrates 4-directional cardinal rotation and quantum density matrix formulations.
"""

import numpy as np
import cv2
from scipy.linalg import sqrtm
from typing import Dict, Any, Tuple, Optional


def extract_sift_density_matrix(image: np.ndarray, max_keypoints: int = 128) -> Dict[str, Any]:
    """
    Extract SIFT keypoint descriptors, map to pure quantum states in 7-qubit space,
    and compute the response-weighted statistical ensemble density matrix rho.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    try:
        detector = cv2.SIFT_create(nfeatures=max_keypoints)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
    except Exception:
        detector = cv2.ORB_create(nfeatures=max_keypoints)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is not None:
            padded = np.zeros((descriptors.shape[0], 128), dtype=np.float32)
            padded[:, :descriptors.shape[1]] = descriptors
            descriptors = padded

    dim = 128
    if descriptors is None or len(descriptors) == 0:
        rho = np.eye(dim, dtype=np.float64) / dim
        return {
            "keypoints": [],
            "descriptors": None,
            "density_matrix": rho,
            "num_kps": 0
        }

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    psi = (descriptors / norms).astype(np.float64)

    responses = np.array([kp.response for kp in keypoints], dtype=np.float64)
    if np.sum(responses) == 0 or np.isnan(np.sum(responses)):
        weights = np.ones(len(keypoints), dtype=np.float64) / len(keypoints)
    else:
        weights = responses / np.sum(responses)

    rho = np.dot(psi.T * weights, psi)
    rho = 0.5 * (rho + rho.T)
    trace = np.trace(rho)
    if trace > 0:
        rho = rho / trace
    else:
        rho = np.eye(dim, dtype=np.float64) / dim

    return {
        "keypoints": keypoints,
        "descriptors": psi,
        "density_matrix": rho,
        "num_kps": len(keypoints)
    }


def compute_von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """Compute Von Neumann entropy S(rho) = - Tr(rho log(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    pos_eig = eigenvalues[eigenvalues > 1e-15]
    if len(pos_eig) == 0:
        return 0.0
    return float(max(0.0, -np.sum(pos_eig * (np.log(pos_eig) / np.log(base)))))


def compute_quantum_jsd(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """Compute Quantum Jensen-Shannon Divergence between two mixed states."""
    rho_mix = 0.5 * (rho_A + rho_B)
    s_mix = compute_von_neumann_entropy(rho_mix, base=2.0)
    s_a = compute_von_neumann_entropy(rho_A, base=2.0)
    s_b = compute_von_neumann_entropy(rho_B, base=2.0)
    return float(max(0.0, s_mix - 0.5 * (s_a + s_b)))


def compute_mixed_state_fidelity(rho_A: np.ndarray, rho_B: np.ndarray) -> float:
    """Compute Uhlmann-Jozsa quantum fidelity F(rho_A, rho_B)."""
    try:
        sqrt_A = sqrtm(rho_A)
        if np.iscomplexobj(sqrt_A):
            sqrt_A = sqrt_A.real
        M = sqrt_A @ rho_B @ sqrt_A
        sqrt_M = sqrtm(M)
        if np.iscomplexobj(sqrt_M):
            sqrt_M = sqrt_M.real
        fidelity = float(np.trace(sqrt_M)) ** 2
        return float(np.clip(fidelity, 0.0, 1.0))
    except Exception:
        hs = np.trace(rho_A @ rho_B)
        return float(np.clip(hs, 0.0, 1.0))


def compute_4direction_sift_match(img_A: np.ndarray, img_B: np.ndarray, max_keypoints: int = 128) -> Dict[str, Any]:
    """
    Performs 4-directional matching between Image A and Image B.
    """
    feat_A = extract_sift_density_matrix(img_A, max_keypoints=max_keypoints)
    rho_A = feat_A["density_matrix"]
    s_A = compute_von_neumann_entropy(rho_A)

    rotations = [
        ("0 deg", None),
        ("90 deg (CW)", cv2.ROTATE_90_CLOCKWISE),
        ("180 deg", cv2.ROTATE_180),
        ("270 deg (CCW)", cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    best_match = {
        "angle_name": None,
        "match_accuracy": -1.0,
        "qjsd": 1.0,
        "fidelity": 0.0
    }
    rotation_details = []

    for angle_name, rot_code in rotations:
        rot_img_B = img_B.copy() if rot_code is None else cv2.rotate(img_B, rot_code)
        feat_B = extract_sift_density_matrix(rot_img_B, max_keypoints=max_keypoints)
        rho_B = feat_B["density_matrix"]
        s_B = compute_von_neumann_entropy(rho_B)

        qjsd = compute_quantum_jsd(rho_A, rho_B)
        fidelity = compute_mixed_state_fidelity(rho_A, rho_B)
        sim_qjsd = max(0.0, (1.0 - np.sqrt(qjsd))) * 100.0
        combined_accuracy = 0.6 * (fidelity * 100.0) + 0.4 * sim_qjsd

        item = {
            "angle": angle_name,
            "kps_B": feat_B["num_kps"],
            "entropy_B": s_B,
            "qjsd": qjsd,
            "fidelity": fidelity,
            "accuracy": combined_accuracy
        }
        rotation_details.append(item)

        if combined_accuracy > best_match["match_accuracy"]:
            best_match["angle_name"] = angle_name
            best_match["match_accuracy"] = combined_accuracy
            best_match["qjsd"] = qjsd
            best_match["fidelity"] = fidelity

    return {
        "entropy_A": s_A,
        "kps_A": feat_A["num_kps"],
        "rotations": rotation_details,
        "best_match": best_match
    }
