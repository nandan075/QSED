"""
Keypoint Descriptor Builder Module.

Constructs:
1. Symmetric Pairwise Entropy Matrix:
       E in R^{n x n},  where E_{ij} = E_{ji} = S_{ij}, and E_{ii} = 0.0
   The diagonal zero is an algorithm storage convention, not a physical claim that
   the individual entropy of an individual keypoint mixed state is zero.
2. (n - 1)-Dimensional Keypoint Descriptors:
   For each keypoint k_i, its descriptor vector is obtained by extracting the i-th row
   of E and removing the diagonal element E_{ii}:
       D(k_i) = [S_{i1}, ..., S_{i, i-1}, S_{i, i+1}, ..., S_{in}] in R^{n - 1}
3. Metadata-Preserving Container:
   Maintains bidirectional links between spatial image coordinates (x, y), detector
   metadata (response, scale, angle), the normalized quantum state |psi_i>, the
   (n - 1)-D descriptor D(k_i), and the full row of E.
4. Edge Case Handling:
   Robust against n = 0 (empty image) and n = 1 (single keypoint, 0-D descriptor) without exceptions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .pairwise_entropy import compute_pairwise_entropy_matrix


@dataclass
class KeypointPairwiseDescriptor:
    """
    Complete descriptor representation for a single keypoint.

    Attributes:
        index: Index i of the keypoint in the keypoint list (0 <= i < n).
        x: Spatial column coordinate in pixels.
        y: Spatial row coordinate in pixels.
        descriptor: (n - 1)-dimensional vector of pairwise entropy values [S_{i, j}]_{j != i}.
        full_entropy_row: Full n-dimensional row E[i, :] including the diagonal 0.0.
        quantum_state: Normalized pure state vector |psi_i> in H_d.
        metadata: Optional dictionary of detector properties (response, scale, angle, etc.).
    """
    index: int
    x: float
    y: float
    descriptor: np.ndarray
    full_entropy_row: np.ndarray
    quantum_state: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pt(self) -> Tuple[float, float]:
        """(x, y) coordinate tuple."""
        return (self.x, self.y)

    @property
    def dim(self) -> int:
        """Dimensionality of the descriptor (n - 1)."""
        return len(self.descriptor)


@dataclass
class ImageEntropyRepresentation:
    """
    Container for the complete pairwise entropy representation of an image.

    Attributes:
        image_shape: (H, W) tuple of image dimensions.
        entropy_matrix: Symmetric matrix E in R^{n x n} with zeros on diagonal.
        descriptors: List of KeypointPairwiseDescriptor objects of length n.
        quantum_states: (n, d) array of all normalized pure states.
        num_keypoints: Total count of keypoints n.
    """
    image_shape: Tuple[int, int]
    entropy_matrix: np.ndarray
    descriptors: List[KeypointPairwiseDescriptor]
    quantum_states: np.ndarray

    @property
    def num_keypoints(self) -> int:
        return len(self.descriptors)

    @property
    def keypoints_xy(self) -> List[Tuple[float, float]]:
        return [desc.pt for desc in self.descriptors]


def build_keypoint_descriptors(
    quantum_states: np.ndarray,
    keypoints_xy: List[Tuple[float, float]],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    image_shape: Tuple[int, int] = (0, 0),
    method: str = "optimized",
    log_base: float = 2.0,
) -> ImageEntropyRepresentation:
    """
    Build the pairwise entropy matrix E and assemble the (n - 1)-dimensional descriptors
    for all n keypoints.

    Handling of Special Cases:
    - n == 0: Returns empty representation (entropy matrix shape (0, 0), 0 descriptors).
    - n == 1: Returns entropy matrix of shape (1, 1) containing [0.0], and a single descriptor
              with an empty array shape (0,) since there are no other keypoints to pair with.
    - n >= 2: Each descriptor D(k_i) has length exactly n - 1.

    Args:
        quantum_states: (n, d) array of normalized pure state vectors.
        keypoints_xy: List of n (x, y) tuples.
        metadata_list: Optional list of n metadata dictionaries.
        image_shape: (H, W) dimensions of the source image.
        method: 'optimized' (subspace eigenvalues) or 'reference' (density matrix diagonalization).
        log_base: Logarithm base for entropy (default: 2.0).

    Returns:
        ImageEntropyRepresentation containing the E matrix and descriptor objects.
    """
    n = len(keypoints_xy)
    if metadata_list is None or len(metadata_list) != n:
        metadata_list = [{} for _ in range(n)]

    # 1. Edge Case: n = 0
    if n == 0:
        return ImageEntropyRepresentation(
            image_shape=image_shape,
            entropy_matrix=np.empty((0, 0), dtype=np.float64),
            descriptors=[],
            quantum_states=np.empty((0, quantum_states.shape[1] if quantum_states.ndim > 1 else 0), dtype=np.float64),
        )

    # 2. Edge Case: n = 1
    if n == 1:
        e_matrix = np.zeros((1, 1), dtype=np.float64)
        x, y = keypoints_xy[0]
        desc = KeypointPairwiseDescriptor(
            index=0,
            x=float(x),
            y=float(y),
            descriptor=np.empty(0, dtype=np.float64),
            full_entropy_row=np.zeros(1, dtype=np.float64),
            quantum_state=quantum_states[0].copy(),
            metadata=metadata_list[0],
        )
        return ImageEntropyRepresentation(
            image_shape=image_shape,
            entropy_matrix=e_matrix,
            descriptors=[desc],
            quantum_states=quantum_states.copy(),
        )

    # 3. Standard Case: n >= 2
    # Compute full symmetric pairwise entropy matrix E
    e_matrix = compute_pairwise_entropy_matrix(quantum_states, method=method, log_base=log_base)

    descriptor_list: List[KeypointPairwiseDescriptor] = []
    for i in range(n):
        x, y = keypoints_xy[i]
        full_row = e_matrix[i, :]

        # Extract (n - 1) values excluding the diagonal entry E_{ii}
        # D(k_i) = [S_{i1}, ..., S_{i, i-1}, S_{i, i+1}, ..., S_{in}]
        desc_vec = np.concatenate([full_row[:i], full_row[i + 1 :]])

        descriptor_obj = KeypointPairwiseDescriptor(
            index=i,
            x=float(x),
            y=float(y),
            descriptor=desc_vec,
            full_entropy_row=full_row.copy(),
            quantum_state=quantum_states[i].copy(),
            metadata=metadata_list[i],
        )
        descriptor_list.append(descriptor_obj)

    return ImageEntropyRepresentation(
        image_shape=image_shape,
        entropy_matrix=e_matrix,
        descriptors=descriptor_list,
        quantum_states=quantum_states,
    )
