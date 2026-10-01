"""
Demonstration and CLI Runner for 24-D Local Von Neumann Entropy Feature Descriptor.

Features:
1. Extract 24-D entropy descriptor for detected keypoints in single images.
2. Inspect and visualize 5x5 neighborhood with P1..P24 labels and P7 local 3x3 neighborhood.
3. Compare two images, match descriptors, and calculate image similarity.
4. Expose sample keypoint descriptors and pairwise Euclidean distances with real numerical values.

Usage:
    # Single image keypoint neighborhood visualization:
    python scripts/run_entropy_descriptor.py --image images/cameraman.png --vis-kp --p-highlight P7

    # Two image similarity and matching:
    python scripts/run_entropy_descriptor.py --image1 images/cameraman.png --image2 images/cameraman.png
    python scripts/run_entropy_descriptor.py --image1 images/cameraman.png --image2 images/house.png
"""

import os
import sys
import argparse
import numpy as np

# Ensure QSED/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classical.image_loader import load_image
from src.feature.entropy_descriptor import (
    extract_entropy_descriptor,
    extract_all_entropy_descriptors,
    compute_image_similarity_entropy_descriptor,
    visualize_keypoint_neighborhood,
    visualize_matches,
    detect_default_keypoints,
    P_LABELS_RASTER,
    P_OFFSETS_DICT,
    get_p_neighborhood_labels
)


def run_single_image_demo(image_path: str, p_highlight: str, output_dir: str):
    print("\n" + "=" * 70)
    print("      24-D LOCAL VON NEUMANN ENTROPY: SINGLE KEYPOINT INSPECTION")
    print("=" * 70)

    image = load_image(image_path, target_size=(256, 256))
    print(f"[*] Loaded image: {image_path} (shape: {image.shape})")

    # Detect keypoints
    kps = detect_default_keypoints(image, max_keypoints=30)
    print(f"[*] Detected {len(kps)} salient keypoints via QSED pipeline.")

    if len(kps) == 0:
        print("[!] No keypoints detected. Exiting.")
        return

    selected_kp = kps[0]
    print(f"\n[*] Selected Keypoint: KP at (x={selected_kp.x}, y={selected_kp.y})")
    print(f"    - Dominant Direction    : {selected_kp.dominant_direction:.1f}°")
    print(f"    - Gradient Magnitude    : {selected_kp.gradient_magnitude:.2f}")
    print(f"    - Directional Variation : {selected_kp.directional_variation:.4f}")
    print(f"    - Keypoint Score        : {selected_kp.keypoint_score:.4f}")

    # Extract 24-D descriptor
    desc = extract_entropy_descriptor(image, selected_kp, ordering="raster", norm="none")
    print(f"\n[*] 24-Dimensional Von Neumann Entropy Descriptor F(KP):")
    print(f"    Shape: {desc.shape} | dtype: {desc.dtype}")
    print("    [" + ", ".join([f"{v:.4f}" for v in desc[:8]]) + " ...")
    print("     " + ", ".join([f"{v:.4f}" for v in desc[8:16]]) + " ...")
    print("     " + ", ".join([f"{v:.4f}" for v in desc[16:]]) + "]")

    # Inspect P7 local 3x3 neighborhood
    print(f"\n[*] Verifying local 3x3 neighborhood for {p_highlight}:")
    p_labels = get_p_neighborhood_labels(p_highlight)
    for row in p_labels:
        print("    " + "  ".join([f"{cell:<5}" for cell in row]))

    # Generate and save visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, f"keypoint_neighborhood_{p_highlight}.png")
    visualize_keypoint_neighborhood(image, selected_kp, p_selected=p_highlight, save_path=vis_path)
    print(f"\n[+] Visualization saved to: {vis_path}")
    print("=" * 70 + "\n")


def run_two_image_demo(img1_path: str, img2_path: str, norm: str, ordering: str, output_dir: str):
    print("\n" + "=" * 70)
    print("      24-D LOCAL VON NEUMANN ENTROPY: TWO-IMAGE SIMILARITY MATCHING")
    print("=" * 70)

    img1 = load_image(img1_path, target_size=(256, 256))
    img2 = load_image(img2_path, target_size=(256, 256))
    print(f"[*] Image 1: {img1_path} (shape: {img1.shape})")
    print(f"[*] Image 2: {img2_path} (shape: {img2.shape})")

    kps1 = detect_default_keypoints(img1, max_keypoints=50)
    kps2 = detect_default_keypoints(img2, max_keypoints=50)
    print(f"[*] Image 1 Keypoints detected : {len(kps1)}")
    print(f"[*] Image 2 Keypoints detected : {len(kps2)}")

    res = compute_image_similarity_entropy_descriptor(
        img1, img2, keypoints_a=kps1, keypoints_b=kps2,
        matching_method="greedy", norm=norm, ordering=ordering
    )

    print(f"[*] Corresponding Matches Found: {res['num_matches']}")
    print(f"[*] Match Coverage Ratio       : {res['match_ratio'] * 100:.2f}%")
    print(f"[*] Mean Descriptor Distance   : {res['mean_descriptor_distance']:.4f}")
    print("-" * 70)
    print(f"[+] Image-Level Similarity Score: {res['similarity_score']:.2f}%")
    print("=" * 70)

    # Show actual sample descriptors and pairwise distance
    if res['num_matches'] > 0:
        first_match = res['raw_matches'][0]
        idx_a, idx_b, pair_dist = first_match
        kp_a = res['keypoints_a'][idx_a]
        kp_b = res['keypoints_b'][idx_b]
        desc_a = res['descriptors_a'][idx_a]
        desc_b = res['descriptors_b'][idx_b]

        print("\n--- Detailed Concrete Example of Matched Keypoints ---")
        print(f"Image A - Keypoint {idx_a + 1} at (x={kp_a.x}, y={kp_a.y}):")
        print("  F(KP_A) = [" + ", ".join([f"{v:.4f}" for v in desc_a[:6]]) + ", ..., " +
              ", ".join([f"{v:.4f}" for v in desc_a[-4:]]) + "]")
        print(f"\nImage B - Keypoint {idx_b + 1} at (x={kp_b.x}, y={kp_b.y}):")
        print("  F(KP_B) = [" + ", ".join([f"{v:.4f}" for v in desc_b[:6]]) + ", ..., " +
              ", ".join([f"{v:.4f}" for v in desc_b[-4:]]) + "]")
        print(f"\nEuclidean Distance between F(KP_A) and F(KP_B):")
        print(f"  ||F(KP_A) - F(KP_B)||_2 = {pair_dist:.4f}")

        print("\n--- Top 10 Matched Keypoint Pairs ---")
        print(f"{'Match':<6} | {'Img1 KP (x,y)':<16} | {'Img2 KP (x,y)':<16} | {'Distance':<10}")
        print("-" * 55)
        for i, (ia, ib, d) in enumerate(res['raw_matches'][:10]):
            ka = res['keypoints_a'][ia]
            kb = res['keypoints_b'][ib]
            print(f"{i+1:<6} | ({ka.x:>3}, {ka.y:>3}){'':<7} | ({kb.x:>3}, {kb.y:>3}){'':<7} | {d:<10.4f}")
        print("-" * 55)

    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, "matches_24d_entropy.png")
    visualize_matches(img1, res['keypoints_a'], img2, res['keypoints_b'], res['raw_matches'], save_path=vis_path)
    print(f"\n[+] Matches visualization saved to: {vis_path}\n")


def main():
    parser = argparse.ArgumentParser(description="24-D Local Von Neumann Entropy Feature Descriptor")
    parser.add_argument("--image", type=str, help="Path to single image for inspection.")
    parser.add_argument("--image1", type=str, help="Path to first image for similarity.")
    parser.add_argument("--image2", type=str, help="Path to second image for similarity.")
    parser.add_argument("--vis-kp", action="store_true", help="Generate 5x5 and 3x3 keypoint neighborhood visualization.")
    parser.add_argument("--p-highlight", type=str, default="P7", help="Pi to highlight in 3x3 visualization (default: P7).")
    parser.add_argument("--norm", type=str, choices=["none", "unit_entropy", "l2"], default="none",
                        help="Normalization strategy (default: 'none').")
    parser.add_argument("--ordering", type=str, choices=["raster", "clockwise"], default="raster",
                        help="Ordering of 24 positions (default: 'raster').")
    parser.add_argument("--output", type=str, default="outputs/entropy_descriptor", help="Output directory.")

    args = parser.parse_args()

    if args.image1 and args.image2:
        run_two_image_demo(args.image1, args.image2, args.norm, args.ordering, args.output)
    elif args.image:
        run_single_image_demo(args.image, args.p_highlight, args.output)
    else:
        # Default demo on cameraman
        default_img = "QSED/images/cameraman.png" if os.path.exists("QSED/images/cameraman.png") else "images/cameraman.png"
        run_single_image_demo(default_img, args.p_highlight, args.output)


if __name__ == "__main__":
    main()
