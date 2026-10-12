"""
Main CLI Pipeline for Pairwise Von Neumann Entropy Keypoint Image Matching.

This script executes the complete end-to-end pipeline:
1. Load Image A and Image B.
2. Detect keypoints (SIFT, ORB, Harris, or Grid).
3. Encode local patches into normalized quantum pure state vectors |psi_i>.
4. Construct symmetric pairwise Von Neumann entropy matrices E^A and E^B.
5. Generate (n_A - 1) and (n_B - 1) relational descriptors D(k_i).
6. Match Image A and Image B using the permutation-aware spectral compatibility algorithm.
7. Perform RANSAC geometric verification.
8. Output numerical metrics, correspondence tables, and visual analysis figures.

IMPORTANT:
Designed for offline execution on another device.
This module does NOT execute automatically on import.
"""

import sys
import os
import time
from typing import Dict, Any, Optional

from .config import build_argument_parser, PairwiseMatchingPipelineConfig
from .image_loader import load_grayscale_image
from .keypoint_extractor import KeypointExtractor
from .quantum_state_encoder import extract_keypoint_quantum_states
from .descriptor_builder import build_keypoint_descriptors, ImageEntropyRepresentation
from .matcher import PairwiseEntropyMatcher, MatchResult
from .visualization import plot_pairwise_entropy_matrices, plot_match_correspondences, plot_single_keypoint_relational_profile


def run_pipeline(
    img1_path: str,
    img2_path: str,
    config: Optional[PairwiseMatchingPipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Execute the Pairwise Von Neumann Entropy Keypoint Matching pipeline.

    Args:
        img1_path: Path to Image A.
        img2_path: Path to Image B.
        config: Optional configuration object. If None, default settings are used.

    Returns:
        Dictionary containing all representations, match results, and execution timings.
    """
    cfg = config or PairwiseMatchingPipelineConfig()
    t_start = time.time()

    if cfg.verbose:
        print("=" * 72)
        print("    PAIRWISE VON NEUMANN ENTROPY KEYPOINT MATCHING PIPELINE")
        print("=" * 72)
        print(f"[*] Image A: {img1_path}")
        print(f"[*] Image B: {img2_path}")
        print(f"[*] Detector: {cfg.keypoint.detector.upper()} (max: {cfg.keypoint.max_keypoints})")
        print(f"[*] Patch Size: {cfg.patch.patch_size}x{cfg.patch.patch_size} | Encoding: {cfg.patch.encoding_method}")
        print(f"[*] Entropy Method: {cfg.entropy.method.upper()} | Sigma: {cfg.matching.entropy_sigma}")
        print(f"[*] RANSAC Model: {cfg.geometric.model.upper()} (thresh: {cfg.geometric.ransac_threshold} px)")
        print("-" * 72)

    # 1. Load Images
    t0 = time.time()
    img_a = load_grayscale_image(img1_path, normalize=False)
    img_b = load_grayscale_image(img2_path, normalize=False)
    if cfg.verbose:
        print(f"[1/5] Loaded images: Image A {img_a.shape}, Image B {img_b.shape} in {time.time() - t0:.3f}s")

    # 2. Keypoint Extraction
    t0 = time.time()
    extractor = KeypointExtractor(
        detector=cfg.keypoint.detector,
        max_keypoints=cfg.keypoint.max_keypoints,
        min_distance=cfg.keypoint.min_distance,
    )
    kps_a, desc_a, meta_a = extractor.extract(img_a)
    kps_b, desc_b, meta_b = extractor.extract(img_b)
    if cfg.verbose:
        print(f"[2/5] Extracted keypoints: Image A = {len(kps_a)}, Image B = {len(kps_b)} in {time.time() - t0:.3f}s")

    # 3. Quantum Pure State Encoding
    t0 = time.time()
    states_a = extract_keypoint_quantum_states(
        img_a,
        kps_a,
        descriptors=desc_a,
        patch_size=cfg.patch.patch_size,
        encoding_method=cfg.patch.encoding_method,
        padding_mode=cfg.patch.padding_mode,
        eps=cfg.patch.epsilon,
    )
    states_b = extract_keypoint_quantum_states(
        img_b,
        kps_b,
        descriptors=desc_b,
        patch_size=cfg.patch.patch_size,
        encoding_method=cfg.patch.encoding_method,
        padding_mode=cfg.patch.padding_mode,
        eps=cfg.patch.epsilon,
    )
    if cfg.verbose:
        dim_a = states_a.shape[1] if len(states_a) > 0 else 0
        print(f"[3/5] Encoded quantum pure states: dim = {dim_a} in {time.time() - t0:.3f}s")

    # 4. Pairwise Von Neumann Entropy Matrix & Descriptors
    t0 = time.time()
    rep_a = build_keypoint_descriptors(
        states_a,
        kps_a,
        metadata_list=meta_a,
        image_shape=img_a.shape,
        method=cfg.entropy.method,
        log_base=cfg.entropy.log_base,
    )
    rep_b = build_keypoint_descriptors(
        states_b,
        kps_b,
        metadata_list=meta_b,
        image_shape=img_b.shape,
        method=cfg.entropy.method,
        log_base=cfg.entropy.log_base,
    )
    if cfg.verbose:
        print(f"[4/5] Built pairwise entropy matrices: E^A {rep_a.entropy_matrix.shape}, E^B {rep_b.entropy_matrix.shape} in {time.time() - t0:.3f}s")

    # 5. Relational Matching & Geometric Verification
    t0 = time.time()
    matcher = PairwiseEntropyMatcher(
        matching_config=cfg.matching,
        geometric_config=cfg.geometric,
    )
    match_result = matcher.match(rep_a, rep_b)
    if cfg.verbose:
        print(f"[5/5] Matched & verified correspondences in {time.time() - t0:.3f}s")

    total_time = time.time() - t_start

    # Summary Report
    if cfg.verbose:
        print("-" * 72)
        print("                  MATCHING PERFORMANCE SUMMARY")
        print("-" * 72)
        print(f"[*] Keypoints Image A       : {match_result.num_kps_a}")
        print(f"[*] Keypoints Image B       : {match_result.num_kps_b}")
        print(f"[*] Initial Candidate Matches: {match_result.num_matches}")
        print(f"[*] RANSAC Verified Inliers : {match_result.inlier_count} ({match_result.geometric_status})")
        print(f"[*] Mean Relational Error   : {match_result.mean_entropy_error:.4f} bits")
        print(f"[*] Overall Similarity Score: {match_result.similarity_score:.2f}%")
        print(f"[*] Total Execution Time    : {total_time:.3f}s")
        print("=" * 72)

        if match_result.num_matches > 0:
            print("\n--- Top 10 Correspondences ---")
            print(f"{'#':<4} | {'KP A (x, y)':<16} | {'KP B (x, y)':<16} | {'Compatibility':<14} | {'Inlier?'}")
            print("-" * 62)
            for k in range(min(10, match_result.num_matches)):
                ia, ib, comp, _ = match_result.matches[k]
                pt_a = f"({rep_a.descriptors[ia].x:.1f}, {rep_a.descriptors[ia].y:.1f})"
                pt_b = f"({rep_b.descriptors[ib].x:.1f}, {rep_b.descriptors[ib].y:.1f})"
                is_inl = "YES" if match_result.inlier_mask[k] else "NO"
                print(f"{k+1:<4} | {pt_a:<16} | {pt_b:<16} | {comp:<14.4f} | {is_inl}")
            print("-" * 62 + "\n")

    # Visualizations
    if cfg.save_visualizations:
        os.makedirs(cfg.output_dir, exist_ok=True)

        # 1. Matches plot
        match_plot_path = os.path.join(cfg.output_dir, "pairwise_entropy_matches.png")
        plot_match_correspondences(img_a, img_b, match_result, save_path=match_plot_path)

        # 2. Entropy heatmaps
        matrix_plot_path = os.path.join(cfg.output_dir, "pairwise_entropy_matrices.png")
        plot_pairwise_entropy_matrices(rep_a, rep_b, save_path=matrix_plot_path)

        # 3. Single keypoint relational profile
        if rep_a.num_keypoints > 0:
            profile_plot_path = os.path.join(cfg.output_dir, "keypoint_relational_profile_a0.png")
            plot_single_keypoint_relational_profile(rep_a, keypoint_index=0, save_path=profile_plot_path)

        if cfg.verbose:
            print(f"[+] Saved visualization figures to: {cfg.output_dir}/\n")

    return {
        "representation_a": rep_a,
        "representation_b": rep_b,
        "match_result": match_result,
        "execution_time_seconds": total_time,
    }


def main():
    """Command-line interface entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    config = PairwiseMatchingPipelineConfig.from_args(args)
    run_pipeline(args.img1, args.img2, config=config)


if __name__ == "__main__":
    main()
