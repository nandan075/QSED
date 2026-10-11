"""
Configuration Module for Pairwise Von Neumann Entropy Keypoint Matching.

Provides strongly typed, dataclass-based configuration schemas and CLI argument
parsers for all algorithmic components:
- Keypoint detection and extraction
- Local patch extraction and quantum state encoding
- Pairwise density matrix and Von Neumann entropy computation
- Candidate correspondence generation and relational matching
- Robust geometric verification (RANSAC)
- Output reporting and visualization
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import argparse
import os


@dataclass
class KeypointConfig:
    """Configuration for keypoint extraction."""
    detector: str = "sift"  # Options: 'sift', 'orb', 'harris', 'qsed_gradient', 'grid'
    max_keypoints: int = 100
    quality_threshold: float = 0.01
    min_distance: float = 4.0


@dataclass
class PatchConfig:
    """Configuration for local image patch extraction and quantum encoding."""
    patch_size: int = 9  # Size of square patch: patch_size x patch_size
    padding_mode: str = "reflect"  # NumPy padding mode: 'reflect' (default), 'edge', 'symmetric'
    encoding_method: str = "patch_amplitude"  # Options: 'patch_amplitude', 'rootsift', 'directional_gradient'
    epsilon: float = 1e-12  # Numerical regularizer for normalization


@dataclass
class EntropyConfig:
    """Configuration for pairwise Von Neumann entropy computation."""
    method: str = "optimized"  # 'optimized' (analytical eigenvalues via overlap) or 'reference' (density matrix diagonalization)
    eigenvalue_tol: float = 1e-15  # Numerical cutoff for non-zero eigenvalues
    log_base: float = 2.0  # Logarithm base for entropy in bits (default: 2.0)
    verify_density_matrix: bool = False  # If True, checks trace=1 and Hermiticity in reference mode


@dataclass
class MatchingConfig:
    """Configuration for relational keypoint matching."""
    seed_method: str = "mutual_nn"  # 'mutual_nn', 'ratio_test', or 'top_k'
    ratio_threshold: float = 0.8  # Lowe's ratio test threshold
    top_k_candidates: int = 5  # Number of candidate matches per keypoint in top_k mode
    entropy_sigma: float = 0.25  # Gaussian kernel scale for relational entropy difference: exp(-|E_ip - E_jq| / sigma)
    relational_weight: float = 0.6  # Weight for relational consistency vs appearance similarity (in [0, 1])
    power_iterations: int = 30  # Number of power iterations for spectral assignment
    ambiguity_threshold: float = 0.15  # Minimum compatibility threshold to retain a match
    one_to_one: bool = True  # Enforce strict 1-to-1 matching constraint


@dataclass
class GeometricConfig:
    """Configuration for RANSAC geometric verification."""
    model: str = "homography"  # 'homography', 'affine', or 'fundamental'
    ransac_threshold: float = 3.0  # Max reprojection/epipolar error in pixels
    max_iterations: int = 2000
    confidence: float = 0.99
    min_inliers: int = 4  # Minimum inlier count required for successful geometric verification


@dataclass
class PairwiseMatchingPipelineConfig:
    """Master configuration bundling all pipeline stages."""
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    geometric: GeometricConfig = field(default_factory=GeometricConfig)
    random_seed: Optional[int] = 42
    output_dir: str = "outputs/pairwise_entropy_matching"
    save_visualizations: bool = True
    verbose: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PairwiseMatchingPipelineConfig":
        """Construct configuration object from parsed command-line arguments."""
        cfg = cls()

        # Keypoints
        if hasattr(args, "detector") and args.detector is not None:
            cfg.keypoint.detector = args.detector
        if hasattr(args, "max_kps") and args.max_kps is not None:
            cfg.keypoint.max_keypoints = args.max_kps

        # Patch & Encoding
        if hasattr(args, "patch_size") and args.patch_size is not None:
            cfg.patch.patch_size = args.patch_size
        if hasattr(args, "encoding") and args.encoding is not None:
            cfg.patch.encoding_method = args.encoding

        # Entropy
        if hasattr(args, "entropy_method") and args.entropy_method is not None:
            cfg.entropy.method = args.entropy_method

        # Matching
        if hasattr(args, "seed_method") and args.seed_method is not None:
            cfg.matching.seed_method = args.seed_method
        if hasattr(args, "entropy_sigma") and args.entropy_sigma is not None:
            cfg.matching.entropy_sigma = args.entropy_sigma
        if hasattr(args, "ambiguity_thresh") and args.ambiguity_thresh is not None:
            cfg.matching.ambiguity_threshold = args.ambiguity_thresh

        # Geometric
        if hasattr(args, "geom_model") and args.geom_model is not None:
            cfg.geometric.model = args.geom_model
        if hasattr(args, "ransac_thresh") and args.ransac_thresh is not None:
            cfg.geometric.ransac_threshold = args.ransac_thresh

        # Output & Miscellaneous
        if hasattr(args, "output_dir") and args.output_dir is not None:
            cfg.output_dir = args.output_dir
        if hasattr(args, "seed") and args.seed is not None:
            cfg.random_seed = args.seed
        if hasattr(args, "quiet") and args.quiet:
            cfg.verbose = False

        return cfg


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        description="Pairwise Von Neumann Entropy Keypoint Image Matching Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input image paths
    parser.add_argument("--img1", type=str, required=True, help="Path to reference Image A.")
    parser.add_argument("--img2", type=str, required=True, help="Path to query Image B.")

    # Keypoint detection
    parser.add_argument(
        "--detector",
        type=str,
        default="sift",
        choices=["sift", "orb", "harris", "qsed_gradient", "grid"],
        help="Keypoint detector algorithm.",
    )
    parser.add_argument("--max-kps", type=int, default=100, help="Maximum keypoints per image.")

    # Patch & Quantum Encoding
    parser.add_argument("--patch-size", type=int, default=9, help="Local patch side length (pixels).")
    parser.add_argument(
        "--encoding",
        type=str,
        default="patch_amplitude",
        choices=["patch_amplitude", "rootsift", "directional_gradient"],
        help="Quantum pure state amplitude encoding method.",
    )

    # Entropy calculation
    parser.add_argument(
        "--entropy-method",
        type=str,
        default="optimized",
        choices=["optimized", "reference"],
        help="Entropy calculation method: 'optimized' (analytical eigenvalues via overlap) or 'reference' (density matrix diagonalization).",
    )

    # Relational Matching
    parser.add_argument(
        "--seed-method",
        type=str,
        default="mutual_nn",
        choices=["mutual_nn", "ratio_test", "top_k"],
        help="Candidate correspondence initialization strategy.",
    )
    parser.add_argument(
        "--entropy-sigma",
        type=float,
        default=0.25,
        help="Kernel bandwidth parameter for pairwise entropy difference.",
    )
    parser.add_argument(
        "--ambiguity-thresh",
        type=float,
        default=0.15,
        help="Compatibility score cutoff for rejecting ambiguous candidate matches.",
    )

    # Geometric Verification
    parser.add_argument(
        "--geom-model",
        type=str,
        default="homography",
        choices=["homography", "affine", "fundamental"],
        help="Geometric verification model for RANSAC.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=3.0,
        help="RANSAC inlier threshold (pixels).",
    )

    # System & Output
    parser.add_argument("--output-dir", type=str, default="outputs/pairwise_entropy_matching", help="Destination folder for outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible RANSAC.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose console logging.")

    return parser
