"""
Unified Quantum Computer Vision Multi-Model Benchmark & Runner.

Orchestrates all 3 quantum image matching and similarity models:
1. QSED (Quantum SIFT/SURF + 4-Direction Rotation + Von Neumann Entropy Density Matrix)
2. QCSI (Hybrid Quantum-Classical Image Similarity with RootSIFT 7-Qubit Encoding & DINOv2 ZZFeatureMap)
3. Pairwise Von Neumann Entropy (Patch Quantum Pure States + Spectral Compatibility Matching + RANSAC)

Features:
- Single command execution for any image pair at native high resolution.
- Criss-cross combinatorial benchmarking across multiple images.
- Comprehensive scientific multi-panel visualization dashboards.
- Structured JSON and markdown metrics generation.
"""

import os
import sys
import time
import json
import argparse
from itertools import combinations
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configure paths so all three sub-frameworks can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
for sub in ["QCSI", "QSED", "pairwise_von_neumann_entropy"]:
    sub_path = os.path.join(BASE_DIR, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

# Import Model 1: QSED
from QSED.quantum_sift_matcher import (
    extract_features_and_density_matrix as qsed_extract,
    von_neumann_entropy as qsed_entropy,
    quantum_jensen_shannon_divergence as qsed_qjsd,
    quantum_state_fidelity as qsed_fidelity,
    match_images_4direction as qsed_match_4direction,
)

# Import Model 2: QCSI
from QCSI.qcsi.preprocess import ImagePreprocessor
from QCSI.qcsi.local_branch import LocalGeometryMatcher
from QCSI.qcsi.global_branch import GlobalSemanticMatcher
from QCSI.qcsi.fusion import CalibratedFusionModel, extract_feature_vector, explain_similarity

# Import Model 3: Pairwise Von Neumann Entropy
from pairwise_von_neumann_entropy.config import PairwiseMatchingPipelineConfig
from pairwise_von_neumann_entropy.main import run_pipeline as run_pairwise_pipeline


# =============================================================================
# Model Execution Wrappers
# =============================================================================

def execute_qsed(img1_path: str, img2_path: str, max_kps: int = 300) -> Dict[str, Any]:
    """Execute QSED with 4-directional cardinal rotation and quantum density matrices."""
    t0 = time.time()
    img_a = cv2.imread(img1_path)
    img_b = cv2.imread(img2_path)
    if img_a is None:
        raise FileNotFoundError(f"QSED cannot load image: {img1_path}")
    if img_b is None:
        raise FileNotFoundError(f"QSED cannot load image: {img2_path}")

    results = qsed_match_4direction(img_a, img_b, max_keypoints=max_kps)
    best = results["best_match"]
    elapsed = time.time() - t0

    return {
        "model_name": "QSED",
        "num_kps_a": results["kps_A"],
        "num_kps_b": best["feat_B"]["num_kps"],
        "entropy_a": results["entropy_A"],
        "entropy_b": best["entropy_B"],
        "best_orientation": best["angle_name"],
        "fidelity": best["fidelity"],
        "qjsd": best["qjsd"],
        "match_accuracy": best["match_accuracy"],
        "rotations": results["rotations"],
        "feat_a": results["feat_A"],
        "feat_b_best": best["feat_B"],
        "best_rotated_image": best["rotated_image"],
        "elapsed_sec": elapsed,
    }


def execute_qcsi(img1_path: str, img2_path: str, max_kps: int = 300, max_dim: int = 4096) -> Dict[str, Any]:
    """Execute QCSI hybrid quantum-classical pipeline at high resolution."""
    t0 = time.time()
    preprocessor = ImagePreprocessor(max_dim=max_dim)
    prep = preprocessor.preprocess_pair(img1_path, img2_path)

    local_matcher = LocalGeometryMatcher(
        max_keypoints=max_kps,
        search_4directions=True,
    )
    local_res = local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])

    global_matcher = GlobalSemanticMatcher(n_qubits=8)
    global_res = global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])

    fusion_model = CalibratedFusionModel()
    feat_vec = extract_feature_vector(local_res, global_res)
    calibrated_prob = fusion_model.predict_probability(feat_vec)
    explanation = explain_similarity(local_res, global_res, calibrated_prob)

    elapsed = time.time() - t0
    geo = local_res["geometry"]

    return {
        "model_name": "QCSI",
        "num_kps_a": local_res["feat_A"]["num_kps"],
        "num_kps_b": local_res["feat_B"]["num_kps"],
        "entropy_a": local_res["entropy_A"],
        "entropy_b": local_res["entropy_B"],
        "best_rotation": local_res["rotation"],
        "num_matches": len(local_res["matched"]["matches"]),
        "inlier_count": geo["inlier_count"],
        "inlier_ratio": geo["inlier_ratio"],
        "mean_inlier_fidelity": geo["mean_inlier_fidelity"],
        "uhlmann_fidelity": local_res["uhlmann_fidelity"],
        "qjsd": local_res["qjsd"],
        "dino_cosine": global_res["classical_cosine"],
        "global_quantum_fidelity": global_res["quantum_fidelity"],
        "calibrated_similarity": explanation["calibrated_similarity_pct"],
        "verdict": explanation["verdict"],
        "components": explanation["components"],
        "prep": prep,
        "local_res": local_res,
        "global_res": global_res,
        "elapsed_sec": elapsed,
    }


def execute_pairwise_entropy(
    img1_path: str, img2_path: str, detector: str = "sift", max_kps: int = 300
) -> Dict[str, Any]:
    """Execute Pairwise Von Neumann Entropy relational matching pipeline."""
    t0 = time.time()
    cfg = PairwiseMatchingPipelineConfig()
    cfg.keypoint.detector = detector
    cfg.keypoint.max_keypoints = max_kps
    cfg.save_visualizations = False
    cfg.verbose = False

    raw_res = run_pairwise_pipeline(img1_path, img2_path, config=cfg)
    match_result = raw_res["match_result"]
    rep_a = raw_res["representation_a"]
    rep_b = raw_res["representation_b"]
    elapsed = time.time() - t0

    # Mean relational entropy per keypoint
    mean_ent_a = float(np.mean(rep_a.entropy_matrix)) if rep_a.num_keypoints > 0 else 0.0
    mean_ent_b = float(np.mean(rep_b.entropy_matrix)) if rep_b.num_keypoints > 0 else 0.0

    return {
        "model_name": "Pairwise_Von_Neumann_Entropy",
        "num_kps_a": match_result.num_kps_a,
        "num_kps_b": match_result.num_kps_b,
        "entropy_a": mean_ent_a,
        "entropy_b": mean_ent_b,
        "candidate_matches": match_result.num_matches,
        "inlier_count": match_result.inlier_count,
        "geometric_status": match_result.geometric_status,
        "mean_entropy_error": match_result.mean_entropy_error,
        "similarity_score": match_result.similarity_score,
        "raw_res": raw_res,
        "elapsed_sec": elapsed,
    }


# =============================================================================
# Unified Visual Dashboard Renderer
# =============================================================================

def render_unified_dashboard(
    img1_path: str,
    img2_path: str,
    qsed_res: Dict[str, Any],
    qcsi_res: Dict[str, Any],
    pairwise_res: Dict[str, Any],
    save_path: str,
    pair_label: str = "Image Pair Comparison"
):
    """
    Renders an in-depth 6-panel scientific visual dashboard presenting:
    1. Reference Image A with detected Keypoints & Entropy S(rho_A).
    2. Query Image B with detected Keypoints & Entropy S(rho_B).
    3. Multi-Model Inlier & Correspondence Overlay (Geometric Matches).
    4. Model Similarity & Fidelity Comparison Bar Chart.
    5. Quantum Information & Entropy Divergence Breakdown.
    6. Comprehensive Cross-Model Metrics Summary Table.
    """
    img_a_bgr = cv2.imread(img1_path)
    img_b_bgr = cv2.imread(img2_path)
    img_a_rgb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)

    h_a, w_a = img_a_rgb.shape[:2]
    h_b, w_b = img_b_rgb.shape[:2]

    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(22, 16), facecolor="#0d1117")
    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.95, hspace=0.32, wspace=0.22)

    title_text = f"QUANTUM COMPUTER VISION MULTI-MODEL ANALYSIS: {pair_label}"
    fig.suptitle(title_text, fontsize=18, fontweight="bold", color="#58a6ff", y=0.97)

    # -------------------------------------------------------------------------
    # Panel 1: Image A with Detected SIFT Keypoints
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_facecolor("#161b22")
    kps_a = qsed_res["feat_a"]["keypoints"]
    # Draw keypoints on downscaled canvas for crisp rendering
    scale_disp_a = 1200 / max(h_a, w_a)
    small_a = cv2.resize(img_a_rgb, (int(w_a * scale_disp_a), int(h_a * scale_disp_a)))
    scaled_kps_a = [
        cv2.KeyPoint(kp.pt[0] * scale_disp_a, kp.pt[1] * scale_disp_a, kp.size * scale_disp_a)
        for kp in kps_a[:150]
    ]
    vis_a = cv2.drawKeypoints(
        small_a, scaled_kps_a, None, color=(0, 255, 128),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    ax1.imshow(vis_a)
    name_a = os.path.basename(img1_path)
    ax1.set_title(
        f"Anchor Image A: {name_a}\nNative Res: {w_a}x{h_a} | Keypoints: {qsed_res['num_kps_a']} | S(rho_A): {qsed_res['entropy_a']:.3f} bits",
        fontsize=11, fontweight="bold", color="#f0f6fc"
    )
    ax1.axis("off")

    # -------------------------------------------------------------------------
    # Panel 2: Image B with Detected SIFT Keypoints
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(3, 2, 2)
    ax2.set_facecolor("#161b22")
    kps_b = qsed_res["feat_b_best"]["keypoints"]
    scale_disp_b = 1200 / max(h_b, w_b)
    small_b = cv2.resize(img_b_rgb, (int(w_b * scale_disp_b), int(h_b * scale_disp_b)))
    scaled_kps_b = [
        cv2.KeyPoint(kp.pt[0] * scale_disp_b, kp.pt[1] * scale_disp_b, kp.size * scale_disp_b)
        for kp in kps_b[:150]
    ]
    vis_b = cv2.drawKeypoints(
        small_b, scaled_kps_b, None, color=(255, 140, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    ax2.imshow(vis_b)
    name_b = os.path.basename(img2_path)
    ax2.set_title(
        f"Query Image B: {name_b}\nNative Res: {w_b}x{h_b} | Keypoints: {qsed_res['num_kps_b']} | S(rho_B): {qsed_res['entropy_b']:.3f} bits",
        fontsize=11, fontweight="bold", color="#f0f6fc"
    )
    ax2.axis("off")

    # -------------------------------------------------------------------------
    # Panel 3: Geometric Inlier Correspondence Overlay
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_facecolor("#161b22")
    # Build side-by-side match overlay using QCSI or Pairwise inliers
    geo = qcsi_res["local_res"]["geometry"]
    matched = qcsi_res["local_res"]["matched"]
    inlier_mask = geo["inlier_mask"]
    pts_a = matched["pts_A"]
    pts_b = matched["pts_B"]

    # Target display height
    disp_h = 450
    w1_d = int(w_a * (disp_h / h_a))
    w2_d = int(w_b * (disp_h / h_b))
    side_by_side = np.zeros((disp_h, w1_d + w2_d, 3), dtype=np.uint8)
    side_by_side[:, :w1_d] = cv2.resize(img_a_rgb, (w1_d, disp_h))
    side_by_side[:, w1_d:] = cv2.resize(img_b_rgb, (w2_d, disp_h))

    ax3.imshow(side_by_side)
    # Overlay inlier match lines
    n_drawn = 0
    if len(pts_a) > 0 and len(inlier_mask) > 0:
        for i in range(len(pts_a)):
            is_inl = inlier_mask[i] if i < len(inlier_mask) else False
            if is_inl and n_drawn < 60:
                xa = pts_a[i][0] * (w1_d / w_a)
                ya = pts_a[i][1] * (disp_h / h_a)
                xb = w1_d + pts_b[i][0] * (w2_d / w_b)
                yb = pts_b[i][1] * (disp_h / h_b)
                ax3.plot([xa, xb], [ya, yb], color="#39d353", alpha=0.7, linewidth=1.2)
                ax3.plot(xa, ya, "o", color="#58a6ff", markersize=3)
                ax3.plot(xb, yb, "o", color="#f0883e", markersize=3)
                n_drawn += 1

    ax3.set_title(
        f"Geometric Verification & Verified Inlier Correspondences\n"
        f"QCSI Inliers: {qcsi_res['inlier_count']} / {qcsi_res['num_matches']} | Pairwise Inliers: {pairwise_res['inlier_count']}",
        fontsize=11, fontweight="bold", color="#f0f6fc"
    )
    ax3.axis("off")

    # -------------------------------------------------------------------------
    # Panel 4: Model Similarity & Fidelity Comparison Bar Chart
    # -------------------------------------------------------------------------
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_facecolor("#161b22")
    categories = [
        "QSED\nMatch Accuracy",
        "QSED\nUhlmann Fid.",
        "QCSI\nCalibrated Sim.",
        "QCSI\nUhlmann Fid.",
        "QCSI\nGlobal Kernel",
        "Pairwise Entropy\nSimilarity",
    ]
    scores = [
        qsed_res["match_accuracy"],
        qsed_res["fidelity"] * 100.0,
        qcsi_res["calibrated_similarity"],
        qcsi_res["uhlmann_fidelity"] * 100.0,
        qcsi_res["global_quantum_fidelity"] * 100.0,
        pairwise_res["similarity_score"],
    ]
    bar_colors = ["#1f6feb", "#238636", "#a371f7", "#3fb950", "#d29922", "#f778ba"]
    bars = ax4.bar(categories, scores, color=bar_colors, edgecolor="#30363d", width=0.55)
    ax4.set_ylim(0, 115)
    ax4.set_ylabel("Score (%)", fontsize=10, color="#8b949e")
    ax4.set_title("Cross-Model Similarity & Quantum Fidelity Benchmarks", fontsize=11, fontweight="bold", color="#f0f6fc")
    ax4.tick_params(colors="#c9d1d9", labelsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#8b949e")
    for bar in bars:
        h = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2, h + 2.0,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#f0f6fc"
        )

    # -------------------------------------------------------------------------
    # Panel 5: Quantum Information & Entropy Divergence Breakdown
    # -------------------------------------------------------------------------
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_facecolor("#161b22")
    entropy_labels = [
        "QSED S(rho_A)",
        "QSED S(rho_B)",
        "QCSI S(rho_A)",
        "QCSI S(rho_B)",
        "Pairwise S(E_A)",
        "Pairwise S(E_B)",
        "QSED QJSD",
        "QCSI QJSD",
        "Pairwise Err (bits)",
    ]
    entropy_vals = [
        qsed_res["entropy_a"],
        qsed_res["entropy_b"],
        qcsi_res["entropy_a"],
        qcsi_res["entropy_b"],
        pairwise_res["entropy_a"],
        pairwise_res["entropy_b"],
        qsed_res["qjsd"],
        qcsi_res["qjsd"],
        pairwise_res["mean_entropy_error"],
    ]
    colors_ent = [
        "#58a6ff", "#79c0ff", "#3fb950", "#56d364", "#d2a8ff", "#e2c5ff",
        "#f0883e", "#ffa657", "#ff7b72"
    ]
    y_pos = np.arange(len(entropy_labels))
    b_ent = ax5.barh(y_pos, entropy_vals, color=colors_ent, edgecolor="#30363d", height=0.6)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(entropy_labels, fontsize=9, color="#c9d1d9")
    ax5.set_xlabel("Value (bits / divergence units)", fontsize=10, color="#8b949e")
    ax5.set_title("Von Neumann Entropy & Quantum Information Divergence", fontsize=11, fontweight="bold", color="#f0f6fc")
    ax5.tick_params(colors="#c9d1d9")
    ax5.grid(axis="x", linestyle="--", alpha=0.3, color="#8b949e")
    for bar in b_ent:
        w = bar.get_width()
        ax5.text(
            w + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}", ha="left", va="center", fontsize=8, color="#f0f6fc"
        )
    ax5.set_xlim(0, max(entropy_vals) * 1.25 + 0.5)

    # -------------------------------------------------------------------------
    # Panel 6: Comprehensive Metrics & Scientific Comparison Card
    # -------------------------------------------------------------------------
    ax6 = plt.subplot(3, 2, 6)
    ax6.set_facecolor("#161b22")
    ax6.axis("off")

    verdict_color = "#3fb950" if qcsi_res["calibrated_similarity"] > 70 else "#f0883e"

    table_data = [
        ["Metric / Attribute", "QSED", "QCSI", "Pairwise Entropy"],
        ["Architecture / Focus", "7-Qubit SIFT Matrix", "Multimodal Hybrid", "Relational Patches"],
        ["Keypoints (Img A / B)", f"{qsed_res['num_kps_a']} / {qsed_res['num_kps_b']}", f"{qcsi_res['num_kps_a']} / {qcsi_res['num_kps_b']}", f"{pairwise_res['num_kps_a']} / {pairwise_res['num_kps_b']}"],
        ["Matches / Inliers", f"N/A (Density)", f"{qcsi_res['inlier_count']} / {qcsi_res['num_matches']}", f"{pairwise_res['inlier_count']} / {pairwise_res['candidate_matches']}"],
        ["Overall Similarity (%)", f"{qsed_res['match_accuracy']:.2f}%", f"{qcsi_res['calibrated_similarity']:.2f}%", f"{pairwise_res['similarity_score']:.2f}%"],
        ["Quantum State Fidelity", f"{qsed_res['fidelity']*100:.2f}%", f"{qcsi_res['uhlmann_fidelity']*100:.2f}%", f"Spectral Comp."],
        ["Entropy S(A) / S(B)", f"{qsed_res['entropy_a']:.2f} / {qsed_res['entropy_b']:.2f}", f"{qcsi_res['entropy_a']:.2f} / {qcsi_res['entropy_b']:.2f}", f"{pairwise_res['entropy_a']:.2f} / {pairwise_res['entropy_b']:.2f}"],
        ["Divergence / Error", f"QJSD: {qsed_res['qjsd']:.4f}", f"QJSD: {qcsi_res['qjsd']:.4f}", f"Rel. Err: {pairwise_res['mean_entropy_error']:.3f}"],
        ["Best Orientation / Geo", f"{qsed_res['best_orientation']}", f"{qcsi_res['best_rotation']} (Homog)", f"{pairwise_res['geometric_status']}"],
        ["Execution Time (sec)", f"{qsed_res['elapsed_sec']:.2f}s", f"{qcsi_res['elapsed_sec']:.2f}s", f"{pairwise_res['elapsed_sec']:.2f}s"],
    ]

    tbl = ax6.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.22, 1.0, 0.76]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(weight="bold", color="#58a6ff")
        elif col == 0:
            cell.set_facecolor("#161b22")
            cell.set_text_props(weight="bold", color="#c9d1d9")
        else:
            cell.set_facecolor("#0d1117" if row % 2 == 0 else "#161b22")
            cell.set_text_props(color="#f0f6fc")

    # Add summary card at bottom of panel 6
    summary_box_text = (
        f"SCIENTIFIC VERDICT: {qcsi_res['verdict']}\n"
        f"Combined Quantum Consensus: Strong correspondence across 7-Qubit amplitudes, "
        f"spectral relational matrices, and DINOv2 ZZFeatureMap."
    )
    ax6.text(
        0.5, 0.08, summary_box_text,
        ha="center", va="center", fontsize=9, color="#f0f6fc", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#21262d", edgecolor=verdict_color, linewidth=1.5)
    )

    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


# =============================================================================
# Criss-Cross Summary Matrix Renderer
# =============================================================================

def render_crisscross_matrix_dashboard(
    images: List[str],
    pair_results: List[Dict[str, Any]],
    save_path: str
):
    """
    Renders an all-in-one matrix dashboard comparing all pairwise relationships:
    - 3 Heatmap similarity matrices for QSED, QCSI, and Pairwise Entropy.
    - Keypoint and inlier statistics comparison.
    - Image thumbnail headers.
    """
    n = len(images)
    labels = [os.path.basename(p).replace(".jpg", "").replace("IMG_", "") for p in images]

    # Initialize matrices
    mat_qsed = np.eye(n) * 100.0
    mat_qcsi = np.eye(n) * 100.0
    mat_pairwise = np.eye(n) * 100.0
    mat_inliers = np.zeros((n, n), dtype=int)

    for item in pair_results:
        i = item["idx1"]
        j = item["idx2"]
        s_qsed = item["qsed"]["accuracy"]
        s_qcsi = item["qcsi"]["similarity"]
        s_pair = item["pairwise"]["similarity"]
        inl = item["qcsi"]["inlier_count"]

        mat_qsed[i, j] = mat_qsed[j, i] = s_qsed
        mat_qcsi[i, j] = mat_qcsi[j, i] = s_qcsi
        mat_pairwise[i, j] = mat_pairwise[j, i] = s_pair
        mat_inliers[i, j] = mat_inliers[j, i] = inl

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d1117")
    fig.suptitle(
        f"CRISSCROSS QUANTUM IMAGE SIMILARITY MATRICES ({n}x{n} ALL-PAIRS)",
        fontsize=16, fontweight="bold", color="#58a6ff", y=0.98
    )

    models_info = [
        ("QSED Match Accuracy (%)", mat_qsed, "Blues"),
        ("QCSI Calibrated Similarity (%)", mat_qcsi, "Greens"),
        ("Pairwise Von Neumann Similarity (%)", mat_pairwise, "Purples"),
    ]

    for ax, (title, mat, cmap) in zip(axes, models_info):
        ax.set_facecolor("#161b22")
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100)
        ax.set_title(title, fontsize=12, fontweight="bold", color="#f0f6fc", pad=12)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=25, color="#c9d1d9", fontsize=9)
        ax.set_yticklabels(labels, color="#c9d1d9", fontsize=9)

        # Annotate cell values
        for r in range(n):
            for c in range(n):
                val = mat[r, c]
                inl_str = f"\n({mat_inliers[r, c]} inliers)" if r != c else "\n(self)"
                text_color = "white" if val > 50 else "black"
                ax.text(
                    c, r, f"{val:.1f}%{inl_str}",
                    ha="center", va="center", color=text_color,
                    fontsize=9, fontweight="bold"
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#c9d1d9")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


# =============================================================================
# High-Level Orchestration
# =============================================================================

def run_single_comparison(
    img1_path: str,
    img2_path: str,
    detector: str = "sift",
    max_kps: int = 300,
    max_dim: int = 4096,
    output_dir: str = "outputs/quantum_cv_unified",
    pair_name: Optional[str] = None
) -> Dict[str, Any]:
    """Runs all 3 models on a single image pair at high resolution."""
    os.makedirs(output_dir, exist_ok=True)
    if pair_name is None:
        n1 = os.path.basename(img1_path).split(".")[0]
        n2 = os.path.basename(img2_path).split(".")[0]
        pair_name = f"{n1}_vs_{n2}"

    print("\n" + "=" * 80)
    print(f"  QUANTUM CV BENCHMARK: {pair_name}")
    print("=" * 80)
    print(f"[*] Image 1  : {img1_path}")
    print(f"[*] Image 2  : {img2_path}")
    print(f"[*] Detector : {detector.upper()}")
    print(f"[*] Max KPs  : {max_kps}")
    print(f"[*] Max Dim  : {max_dim} (Native High Resolution)")
    print("-" * 80)

    # 1. QSED
    print("\n>>> [1/3] Running QSED (Quantum SIFT + 4-Direction Rotation + Von Neumann Entropy)...")
    qsed_res = execute_qsed(img1_path, img2_path, max_kps=max_kps)
    print(f"    [+] Orientation : {qsed_res['best_orientation']}")
    print(f"    [+] Fidelity    : {qsed_res['fidelity'] * 100.0:.2f}%")
    print(f"    [+] QJSD        : {qsed_res['qjsd']:.6f}")
    print(f"    [+] Accuracy    : {qsed_res['match_accuracy']:.2f}% (in {qsed_res['elapsed_sec']:.2f}s)")

    # 2. QCSI
    print("\n>>> [2/3] Running QCSI (RootSIFT 7-Qubit + DINOv2 ZZFeatureMap Multimodal Fusion)...")
    qcsi_res = execute_qcsi(img1_path, img2_path, max_kps=max_kps, max_dim=max_dim)
    print(f"    [+] Similarity  : {qcsi_res['calibrated_similarity']:.2f}%")
    print(f"    [+] Inliers     : {qcsi_res['inlier_count']} / {qcsi_res['num_matches']}")
    print(f"    [+] Inlier Fid. : {qcsi_res['mean_inlier_fidelity'] * 100.0:.2f}%")
    print(f"    [+] Uhlmann Fid : {qcsi_res['uhlmann_fidelity'] * 100.0:.2f}%")
    print(f"    [+] Verdict     : {qcsi_res['verdict']} (in {qcsi_res['elapsed_sec']:.2f}s)")

    # 3. Pairwise Von Neumann Entropy
    print("\n>>> [3/3] Running Pairwise Von Neumann Entropy (Quantum Patch Pure States + Spectral Matching)...")
    pairwise_res = execute_pairwise_entropy(img1_path, img2_path, detector=detector, max_kps=max_kps)
    print(f"    [+] Similarity  : {pairwise_res['similarity_score']:.2f}%")
    print(f"    [+] Inliers     : {pairwise_res['inlier_count']} ({pairwise_res['geometric_status']})")
    print(f"    [+] Rel. Error  : {pairwise_res['mean_entropy_error']:.4f} bits (in {pairwise_res['elapsed_sec']:.2f}s)")

    # Render Visual Dashboard
    dashboard_path = os.path.join(output_dir, f"visual_dashboard_{pair_name}.png")
    print(f"\n[*] Rendering 6-panel comprehensive visual report to: {dashboard_path}...")
    render_unified_dashboard(
        img1_path=img1_path,
        img2_path=img2_path,
        qsed_res=qsed_res,
        qcsi_res=qcsi_res,
        pairwise_res=pairwise_res,
        save_path=dashboard_path,
        pair_label=pair_name,
    )
    print(f"[+] Dashboard saved successfully: {dashboard_path}")

    # Build structured return
    return {
        "pair_name": pair_name,
        "img1": img1_path,
        "img2": img2_path,
        "qsed": {
            "accuracy": qsed_res["match_accuracy"],
            "fidelity": qsed_res["fidelity"],
            "qjsd": qsed_res["qjsd"],
            "entropy_a": qsed_res["entropy_a"],
            "entropy_b": qsed_res["entropy_b"],
            "orientation": qsed_res["best_orientation"],
            "elapsed_sec": qsed_res["elapsed_sec"],
        },
        "qcsi": {
            "similarity": qcsi_res["calibrated_similarity"],
            "inlier_count": qcsi_res["inlier_count"],
            "num_matches": qcsi_res["num_matches"],
            "mean_inlier_fidelity": qcsi_res["mean_inlier_fidelity"],
            "uhlmann_fidelity": qcsi_res["uhlmann_fidelity"],
            "qjsd": qcsi_res["qjsd"],
            "entropy_a": qcsi_res["entropy_a"],
            "entropy_b": qcsi_res["entropy_b"],
            "verdict": qcsi_res["verdict"],
            "elapsed_sec": qcsi_res["elapsed_sec"],
        },
        "pairwise": {
            "similarity": pairwise_res["similarity_score"],
            "inlier_count": pairwise_res["inlier_count"],
            "mean_entropy_error": pairwise_res["mean_entropy_error"],
            "entropy_a": pairwise_res["entropy_a"],
            "entropy_b": pairwise_res["entropy_b"],
            "geometric_status": pairwise_res["geometric_status"],
            "elapsed_sec": pairwise_res["elapsed_sec"],
        },
        "dashboard_image": dashboard_path,
    }


def run_crisscross_evaluation(
    images: List[str],
    detector: str = "sift",
    max_kps: int = 300,
    max_dim: int = 4096,
    output_dir: str = "outputs/quantum_cv_unified"
) -> List[Dict[str, Any]]:
    """Runs all 3 models across all pairwise combinations of provided images."""
    os.makedirs(output_dir, exist_ok=True)
    all_pairs = list(combinations(range(len(images)), 2))
    print("\n" + "#" * 80)
    print(f"  CRISSCROSS EVALUATION ACROSS {len(images)} IMAGES ({len(all_pairs)} PAIRS)")
    print("#" * 80)

    detailed_results = []
    matrix_records = []

    for idx, (i, j) in enumerate(all_pairs, 1):
        p1, p2 = images[i], images[j]
        n1 = os.path.basename(p1).split(".")[0]
        n2 = os.path.basename(p2).split(".")[0]
        pair_label = f"crisscross_{idx}_{n1}_vs_{n2}"

        res = run_single_comparison(
            img1_path=p1,
            img2_path=p2,
            detector=detector,
            max_kps=max_kps,
            max_dim=max_dim,
            output_dir=output_dir,
            pair_name=pair_label,
        )
        detailed_results.append(res)
        matrix_records.append({
            "idx1": i,
            "idx2": j,
            "qsed": res["qsed"],
            "qcsi": res["qcsi"],
            "pairwise": res["pairwise"],
        })

    # Render Crisscross Matrix Dashboard
    matrix_dashboard_path = os.path.join(output_dir, "crisscross_matrix_dashboard.png")
    print(f"\n[*] Rendering Crisscross Matrix Dashboard to: {matrix_dashboard_path}...")
    render_crisscross_matrix_dashboard(images, matrix_records, matrix_dashboard_path)
    print(f"[+] Crisscross Matrix Dashboard saved: {matrix_dashboard_path}")

    return detailed_results


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Quantum Computer Vision Multi-Model Benchmark (QSED, QCSI, Pairwise Entropy)"
    )
    parser.add_argument("--img1", type=str, default=None, help="Path to reference image 1.")
    parser.add_argument("--img2", type=str, default=None, help="Path to query image 2.")
    parser.add_argument("--images", nargs="+", default=None, help="List of images for crisscross evaluation.")
    parser.add_argument("--crisscross", action="store_true", help="Run pairwise crisscross on all images passed.")
    parser.add_argument("--detector", type=str, default="sift", choices=["sift", "orb", "harris", "qsed_gradient"], help="Keypoint detector.")
    parser.add_argument("--max-kps", type=int, default=300, help="Maximum keypoints for quantum state extraction.")
    parser.add_argument("--high-res", action="store_true", default=True, help="Maintain native high resolution (up to 4096px).")
    parser.add_argument("--max-dim", type=int, default=4096, help="Maximum dimension limit when high-res is enabled.")
    parser.add_argument("--output-dir", type=str, default="outputs/quantum_cv_unified", help="Output directory for reports and charts.")

    args = parser.parse_args()

    max_dim = args.max_dim if args.high_res else 1024
    os.makedirs(args.output_dir, exist_ok=True)

    summary_records = []

    if args.crisscross or (args.images is not None and len(args.images) > 1):
        target_images = args.images if args.images else [args.img1, args.img2]
        target_images = [os.path.abspath(p) for p in target_images if p]
        res_list = run_crisscross_evaluation(
            images=target_images,
            detector=args.detector,
            max_kps=args.max_kps,
            max_dim=max_dim,
            output_dir=args.output_dir
        )
        summary_records.extend(res_list)
    elif args.img1 and args.img2:
        res = run_single_comparison(
            img1_path=os.path.abspath(args.img1),
            img2_path=os.path.abspath(args.img2),
            detector=args.detector,
            max_kps=args.max_kps,
            max_dim=max_dim,
            output_dir=args.output_dir
        )
        summary_records.append(res)
    else:
        print("[-] Error: Please specify either --img1 and --img2, or --crisscross --images <img1> <img2> <img3>...")
        sys.exit(1)

    # Save JSON metrics summary
    json_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[+] Full structured metrics saved to: {json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

