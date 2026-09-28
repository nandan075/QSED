"""
QCSI: Hybrid Quantum-Classical Image Similarity CLI Demo & Verification Runner.

Runs end-to-end evaluation on image pairs or generated geometric demonstrations,
prints a detailed terminal report, and saves a 6-panel scientific dashboard.
"""

from typing import Optional
import os
import sys
import argparse
import numpy as np
import cv2

# Ensure local qcsi package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qcsi.preprocess import ImagePreprocessor
from qcsi.local_branch import LocalGeometryMatcher
from qcsi.global_branch import GlobalSemanticMatcher
from qcsi.fusion import CalibratedFusionModel, extract_feature_vector, explain_similarity
from experiments.plots import plot_comprehensive_dashboard
from data.synthetic_generator import SyntheticImagePerturber


def create_demo_pair(output_dir: str):
    """Generates a rich photographic test pair with known geometric transformation."""
    os.makedirs(output_dir, exist_ok=True)
    img_A = np.zeros((400, 400, 3), dtype=np.uint8) + 240

    # Draw shapes, textures, and landmarks
    cv2.circle(img_A, (120, 120), 55, (200, 50, 50), -1)
    cv2.rectangle(img_A, (220, 90), (330, 200), (40, 180, 50), -1)
    pts = np.array([[200, 250], [100, 350], [300, 350]], np.int32)
    cv2.fillPoly(img_A, [pts], (50, 60, 220))
    cv2.putText(img_A, "QCSI HYBRID", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    perturber = SyntheticImagePerturber(seed=42)
    img_B = perturber.rotate(img_A, 25.0)
    img_B = perturber.scale_and_crop(img_B, 1.1)
    img_B = perturber.adjust_photometry(img_B, alpha=1.1, beta=10)

    pA = os.path.join(output_dir, "demo_anchor.png")
    pB = os.path.join(output_dir, "demo_transformed.png")
    cv2.imwrite(pA, cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR))
    cv2.imwrite(pB, cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR))
    return pA, pB


def main():
    parser = argparse.ArgumentParser(
        description="QCSI Hybrid Quantum-Classical Image Similarity Verification."
    )
    parser.add_argument("--img1", type=str, default=None, help="Path to reference image A.")
    parser.add_argument("--img2", type=str, default=None, help="Path to query image B.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic test pair and run verification.")
    parser.add_argument("--save_plot", type=str, default="demo_report.png", help="Path to save 6-panel verification report.")
    parser.add_argument("--max_kps", type=int, default=500, help="Maximum local keypoints for 7-qubit encoding.")
    parser.add_argument("--n_qubits", type=int, default=8, help="Number of qubits for ZZFeatureMap.")
    parser.add_argument("--search_4dir", action="store_true", help="Enable 4-direction cardinal rotation search.")

    args = parser.parse_args()

    if args.demo or (args.img1 is None and args.img2 is None):
        print("\n[+] Initializing QCSI in DEMO mode with generated geometric test pair...")
        demo_dir = os.path.join(current_dir, "demo_data")
        img1_path, img2_path = create_demo_pair(demo_dir)
    else:
        img1_path, img2_path = args.img1, args.img2

    if not os.path.exists(img1_path):
        print(f"[-] Error: Image path '{img1_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(img2_path):
        print(f"[-] Error: Image path '{img2_path}' does not exist.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("        QCSI: HYBRID QUANTUM-CLASSICAL IMAGE SIMILARITY PIPELINE")
    print("=" * 80)
    print(f"Image A (Reference) : {img1_path}")
    print(f"Image B (Query)     : {img2_path}")
    print(f"Local Hilbert Space : 7 Qubits (dim=128)")
    print(f"Global Feature Map  : {args.n_qubits} Qubits ZZFeatureMap\n")

    # 1. Preprocess
    print("[*] Stage 1: Preprocessing (EXIF fix, CLAHE, aspect-ratio scaling)...")
    preprocessor = ImagePreprocessor()
    prep = preprocessor.preprocess_pair(img1_path, img2_path)

    # 2. Local Branch A
    print("[*] Stage 2: Branch A - RootSIFT, 7-Qubit Amplitude Encoding, Trace Distance & RANSAC...")
    local_matcher = LocalGeometryMatcher(
        max_keypoints=args.max_kps,
        search_4directions=args.search_4dir,
    )
    local_res = local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])

    # 3. Global Branch B
    print("[*] Stage 3: Branch B - DINOv2 ViT-B/14, PCA Projection & ZZFeatureMap Quantum Kernel...")
    global_matcher = GlobalSemanticMatcher(n_qubits=args.n_qubits)
    global_res = global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])

    # 4. Multimodal Fusion & Calibration
    print("[*] Stage 4: Multimodal Feature Vector Fusion & Probability Calibration...")
    fusion_model = CalibratedFusionModel()
    feat_vec = extract_feature_vector(local_res, global_res)
    calibrated_prob = fusion_model.predict_probability(feat_vec)
    explanation = explain_similarity(local_res, global_res, calibrated_prob)

    # Print Formatted Results
    geo = local_res["geometry"]
    ev = explanation["evidence"]
    comps = explanation["components"]

    print("\n" + "-" * 80)
    print("                      QCSI DETAILED VERIFICATION RESULTS")
    print("-" * 80)
    print(f"  • CALIBRATED SIMILARITY      : {explanation['calibrated_similarity_pct']:.2f}%")
    print(f"  • VERDICT                    : {explanation['verdict']}")
    print(f"  • RANSAC Inlier Matches      : {geo['inlier_count']} / {len(local_res['matched']['matches'])}")
    print(f"  • Mean Inlier Quantum Fid. K : {geo['mean_inlier_fidelity'] * 100.0:.2f}%")
    print(f"  • Mean Quantum Trace Dist. D : {geo['mean_trace_distance']:.4f}")
    print(f"  • Uhlmann Mixed State Fid. F : {local_res['uhlmann_fidelity'] * 100.0:.2f}%")
    print(f"  • Quantum JSD Divergence     : {local_res['qjsd']:.6f}")
    print(f"  • Von Neumann Entropy A / B  : {local_res['entropy_A']:.3f} / {local_res['entropy_B']:.3f} bits")
    print(f"  • Classical DINOv2 Cosine    : {global_res['classical_cosine']:.4f}")
    print(f"  • Global ZZFeatureMap Fid.   : {global_res['quantum_fidelity'] * 100.0:.2f}%")
    print("-" * 80)
    print("COMPONENT SCORE BREAKDOWN:")
    print(f"  [1] Geometric Structure Score : {comps['geometric_score']:.1f}%")
    print(f"  [2] Semantic Content Score    : {comps['semantic_score']:.1f}%")
    print(f"  [3] Quantum Information Score : {comps['quantum_state_score']:.1f}%")
    print("-" * 80)
    print(f"REASONING:\n  {explanation['reasoning']}")
    print("=" * 80 + "\n")

    # 5. Visual Dashboard
    plot_path = os.path.join(current_dir, args.save_plot) if not os.path.isabs(args.save_plot) else args.save_plot
    print(f"[*] Rendering 6-panel comprehensive visual report to: {plot_path}...")
    plot_comprehensive_dashboard(
        img_A=prep["rgb_A"],
        img_B=prep["rgb_B"],
        local_result=local_res,
        global_result=global_res,
        explanation=explanation,
        save_path=plot_path,
    )
    print(f"[+] Visual report successfully saved: {plot_path}\n")


if __name__ == "__main__":
    main()
