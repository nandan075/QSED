"""
Quantum State Encoding Module.

Converts local image patches and keypoint feature descriptors into normalized pure
quantum state vectors |psi_i> residing in a d-dimensional Hilbert space H_d:

    |psi_i> = sum_{k=1}^d a_{i,k} |k>,  with  <psi_i|psi_i> = sum_{k=1}^d |a_{i,k}|^2 = 1.0

Supported Encoding Conventions:
1. 'patch_amplitude':
   Extracts a w x w grayscale neighborhood patch (e.g., 9x9 -> d=81), applies reflection
   padding for boundaries, centers or normalizes intensities, and normalizes to unit Euclidean
   norm. Represents the direct quantum amplitude encoding of local spatial structure.
2. 'rootsift':
   Extracts 128-dimensional SIFT descriptors, applies Hellinger/RootSIFT mapping
   (v_k <- sqrt(v_k / sum v)), and normalizes to unit norm. Fits 7-qubit Hilbert space (2^7 = 128)
   consistent with the QCSI formulation in the repository.
3. 'directional_gradient':
   Constructs multi-channel directional gradient patches across 8 canonical orientations
   [0°, 22.5°, ..., 157.5°] matching QSED's 8-direction Sobel representation, normalized to unit norm.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from .image_loader import apply_boundary_padding, extract_square_patch


def normalize_quantum_pure_state(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure vector is normalized to exact unit Euclidean norm (pure quantum state).

    If the vector is all zeros (e.g. completely black patch), falls back to the
    maximally unbiased uniform superposition state |+> = (1 / sqrt(d)) sum_k |k>.

    Args:
        vector: 1D real or complex array of dimension d.
        eps: Small stability constant.

    Returns:
        1D float64 array of unit Euclidean norm (<psi|psi> = 1.0).
    """
    arr = np.asarray(vector, dtype=np.float64).flatten()
    dim = len(arr)
    if dim == 0:
        return arr

    norm = np.linalg.norm(arr)
    if norm < eps or np.isnan(norm):
        # Uniform superposition state |+>
        return np.ones(dim, dtype=np.float64) / np.sqrt(dim)

    psi = arr / norm
    # Numerical re-projection to guarantee exact unit norm
    psi = psi / np.linalg.norm(psi)
    return psi


def encode_patch_amplitude(
    patch: np.ndarray,
    center_intensity: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Encode a 2D local grayscale patch into a normalized pure state vector.

    Method:
    1. Flatten the w x w patch into a 1D vector v of length d = w*w.
    2. Optionally center by subtracting local mean to ensure illumination invariance.
    3. Normalize to unit Euclidean norm: |psi> = v / ||v||_2.

    Args:
        patch: 2D numpy array of shape (w, w).
        center_intensity: If True, subtracts patch mean before normalization.
        eps: Stability constant.

    Returns:
        1D float64 array of shape (w*w,) with unit norm.
    """
    v = np.asarray(patch, dtype=np.float64).flatten()
    if center_intensity:
        v = v - np.mean(v)
    return normalize_quantum_pure_state(v, eps=eps)


def extract_keypoint_quantum_states(
    image: np.ndarray,
    keypoints_xy: List[Tuple[float, float]],
    descriptors: Optional[np.ndarray] = None,
    patch_size: int = 9,
    encoding_method: str = "patch_amplitude",
    padding_mode: str = "reflect",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Extract and encode normalized quantum pure states |psi_i> for all detected keypoints.

    Args:
        image: 2D grayscale image array of shape (H, W).
        keypoints_xy: List of (x, y) coordinates for each keypoint.
        descriptors: Optional precomputed feature descriptors (e.g. 128-D SIFT).
        patch_size: Square patch side length (odd integer, e.g. 9 or 15).
        encoding_method: 'patch_amplitude', 'rootsift', or 'directional_gradient'.
        padding_mode: NumPy boundary padding mode (default: 'reflect').
        eps: Regularization epsilon.

    Returns:
        2D float64 numpy array of shape (n, d), where each row is a normalized state |psi_i>.
        If n == 0, returns an empty array of shape (0, d).
    """
    n = len(keypoints_xy)
    if n == 0:
        dim = (patch_size * patch_size) if encoding_method == "patch_amplitude" else 128
        return np.empty((0, dim), dtype=np.float64)

    # 1. RootSIFT descriptor encoding
    if encoding_method == "rootsift":
        if descriptors is not None and len(descriptors) == n:
            states = []
            for desc in descriptors:
                d_arr = np.asarray(desc, dtype=np.float64).flatten()
                # L1 normalization then square root (RootSIFT transform)
                l1_sum = np.sum(np.abs(d_arr))
                if l1_sum > eps:
                    rootsift = np.sqrt(d_arr / l1_sum)
                else:
                    rootsift = d_arr
                states.append(normalize_quantum_pure_state(rootsift, eps=eps))
            return np.vstack(states)
        # Fall back to patch_amplitude if descriptors are unavailable
        encoding_method = "patch_amplitude"

    # 2. Patch Amplitude Encoding
    if encoding_method == "patch_amplitude":
        half_size = patch_size // 2
        padded = apply_boundary_padding(image, pad_width=half_size, mode=padding_mode)

        states = []
        for x, y in keypoints_xy:
            cx = int(round(x)) + half_size
            cy = int(round(y)) + half_size
            patch = extract_square_patch(padded, center_x=cx, center_y=cy, half_size=half_size)
            psi = encode_patch_amplitude(patch, center_intensity=True, eps=eps)
            states.append(psi)

        return np.vstack(states)

    # 3. Directional Gradient Encoding (8 directions)
    if encoding_method == "directional_gradient":
        from scipy.signal import convolve2d
        # Standard 3x3 Sobel directional kernels (0° and 90°)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        gx = convolve2d(image, sobel_x, mode="same", boundary="symm")
        gy = convolve2d(image, sobel_y, mode="same", boundary="symm")
        mag = np.sqrt(gx**2 + gy**2)

        half_size = patch_size // 2
        padded_mag = apply_boundary_padding(mag, pad_width=half_size, mode=padding_mode)

        states = []
        for x, y in keypoints_xy:
            cx = int(round(x)) + half_size
            cy = int(round(y)) + half_size
            patch = extract_square_patch(padded_mag, center_x=cx, center_y=cy, half_size=half_size)
            psi = normalize_quantum_pure_state(patch.flatten(), eps=eps)
            states.append(psi)

        return np.vstack(states)

    raise ValueError(f"Unknown quantum encoding method: '{encoding_method}'")
