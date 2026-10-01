"""
Von Neumann entropy keypoint detection extension.
Exports all public functions/classes from the feature modules.
"""
from .directional_variation import angular_difference, compute_directional_variation, compute_keypoint_score
from .neighborhood import extract_neighborhood, extract_feature_neighborhood
from .density_matrix import build_intensity_density_matrix, build_covariance_density_matrix, verify_density_matrix
from .von_neumann import von_neumann_entropy, normalized_entropy, compute_entropy_map, save_entropy_outputs
from .keypoint_detection import Keypoint, detect_keypoints, keypoints_to_csv, keypoints_to_array
from .ranking import rank_keypoints, select_top_keypoints
from .similarity import (
    compute_similarity,
    compute_entropy_deviation,
    compute_entropy_complexity_similarity,
    compute_spatial_pyramid_entropy_similarity,
    compute_pooled_spatial_correlation,
    save_similarity_outputs,
)
from .quantum_sift_entropy import (
    extract_sift_density_matrix,
    compute_von_neumann_entropy,
    compute_quantum_jsd,
    compute_mixed_state_fidelity,
    compute_4direction_sift_match
)
from .entropy_descriptor import (
    extract_entropy_descriptor,
    extract_all_entropy_descriptors,
    match_entropy_descriptors,
    compute_pairwise_distances,
    compute_image_similarity_entropy_descriptor,
    apply_reflection_padding,
    get_p_neighborhood_labels,
    visualize_keypoint_neighborhood,
    visualize_matches,
    P_LABELS_RASTER,
    P_LABELS_CLOCKWISE,
    P_OFFSETS_DICT,
)

__all__ = [
    'angular_difference',
    'compute_directional_variation',
    'compute_keypoint_score',
    'extract_neighborhood',
    'extract_feature_neighborhood',
    'build_intensity_density_matrix',
    'build_covariance_density_matrix',
    'verify_density_matrix',
    'von_neumann_entropy',
    'normalized_entropy',
    'compute_entropy_map',
    'save_entropy_outputs',
    'Keypoint',
    'detect_keypoints',
    'keypoints_to_csv',
    'keypoints_to_array',
    'rank_keypoints',
    'select_top_keypoints',
    'compute_similarity',
    'compute_entropy_deviation',
    'compute_entropy_complexity_similarity',
    'compute_spatial_pyramid_entropy_similarity',
    'compute_pooled_spatial_correlation',
    'save_similarity_outputs',
    'extract_sift_density_matrix',
    'compute_von_neumann_entropy',
    'compute_quantum_jsd',
    'compute_mixed_state_fidelity',
    'compute_4direction_sift_match',
    'extract_entropy_descriptor',
    'extract_all_entropy_descriptors',
    'match_entropy_descriptors',
    'compute_pairwise_distances',
    'compute_image_similarity_entropy_descriptor',
    'apply_reflection_padding',
    'get_p_neighborhood_labels',
    'visualize_keypoint_neighborhood',
    'visualize_matches',
    'P_LABELS_RASTER',
    'P_LABELS_CLOCKWISE',
    'P_OFFSETS_DICT',
]

