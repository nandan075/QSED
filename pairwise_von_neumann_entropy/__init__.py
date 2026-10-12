"""
Pairwise Von Neumann Entropy Keypoint Image Matching Module.

Exports:
- Configuration schemas: PairwiseMatchingPipelineConfig, MatchingConfig, GeometricConfig, etc.
- Image preprocessing: load_grayscale_image, apply_boundary_padding, generate_synthetic_test_image.
- Quantum state encoding: normalize_quantum_pure_state, encode_patch_amplitude, extract_keypoint_quantum_states.
- Pairwise entropy: construct_pairwise_density_matrix, compute_pairwise_entropy_reference, compute_pairwise_entropy_optimized, compute_pairwise_entropy_matrix, verify_density_matrix.
- Descriptor building: KeypointPairwiseDescriptor, ImageEntropyRepresentation, build_keypoint_descriptors.
- Keypoint matching: PairwiseEntropyMatcher, MatchResult.
- Geometric verification: verify_matches_ransac.
- Visualization: plot_match_correspondences, plot_pairwise_entropy_matrices, plot_single_keypoint_relational_profile.
- Main execution: run_pipeline.
"""

from .config import (
    PairwiseMatchingPipelineConfig,
    KeypointConfig,
    PatchConfig,
    EntropyConfig,
    MatchingConfig,
    GeometricConfig,
    build_argument_parser,
)
from .image_loader import (
    load_grayscale_image,
    apply_boundary_padding,
    extract_square_patch,
    generate_synthetic_test_image,
)
from .keypoint_extractor import (
    KeypointExtractor,
    detect_harris_corners_numpy,
    detect_grid_keypoints,
)
from .quantum_state_encoder import (
    normalize_quantum_pure_state,
    encode_patch_amplitude,
    extract_keypoint_quantum_states,
)
from .pairwise_entropy import (
    construct_pairwise_density_matrix,
    verify_density_matrix,
    compute_pairwise_entropy_reference,
    compute_pairwise_entropy_optimized,
    compute_pairwise_entropy_matrix_optimized,
    compute_pairwise_entropy_matrix_reference,
    compute_pairwise_entropy_matrix,
)
from .descriptor_builder import (
    KeypointPairwiseDescriptor,
    ImageEntropyRepresentation,
    build_keypoint_descriptors,
)
from .geometric_verification import (
    verify_matches_ransac,
    fit_homography_dlt,
    ransac_homography_numpy,
)
from .matcher import (
    MatchResult,
    generate_candidate_seeds,
    build_compatibility_matrix,
    solve_spectral_assignment,
    PairwiseEntropyMatcher,
)
from .visualization import (
    plot_pairwise_entropy_matrices,
    plot_match_correspondences,
    plot_single_keypoint_relational_profile,
)
from .main import run_pipeline

__all__ = [
    "PairwiseMatchingPipelineConfig",
    "KeypointConfig",
    "PatchConfig",
    "EntropyConfig",
    "MatchingConfig",
    "GeometricConfig",
    "build_argument_parser",
    "load_grayscale_image",
    "apply_boundary_padding",
    "extract_square_patch",
    "generate_synthetic_test_image",
    "KeypointExtractor",
    "detect_harris_corners_numpy",
    "detect_grid_keypoints",
    "normalize_quantum_pure_state",
    "encode_patch_amplitude",
    "extract_keypoint_quantum_states",
    "construct_pairwise_density_matrix",
    "verify_density_matrix",
    "compute_pairwise_entropy_reference",
    "compute_pairwise_entropy_optimized",
    "compute_pairwise_entropy_matrix_optimized",
    "compute_pairwise_entropy_matrix_reference",
    "compute_pairwise_entropy_matrix",
    "KeypointPairwiseDescriptor",
    "ImageEntropyRepresentation",
    "build_keypoint_descriptors",
    "verify_matches_ransac",
    "fit_homography_dlt",
    "ransac_homography_numpy",
    "MatchResult",
    "generate_candidate_seeds",
    "build_compatibility_matrix",
    "solve_spectral_assignment",
    "PairwiseEntropyMatcher",
    "plot_pairwise_entropy_matrices",
    "plot_match_correspondences",
    "plot_single_keypoint_relational_profile",
    "run_pipeline",
]
