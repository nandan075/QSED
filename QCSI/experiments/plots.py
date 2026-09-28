"""
QCSI Plotting and Scientific Visualization Suite.

Generates:
1. Reliability Diagrams (Calibration Curves) with ECE.
2. ROC and Precision-Recall Curves across Ablations.
3. Robustness Curves (Score vs Rotation Angle, Scale, Noise, JPEG Quality).
4. Comprehensive 6-Panel Pairwise Verification Dashboard.
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Reliability Diagram (Calibration Quality)",
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Renders a reliability diagram comparing mean predicted probability against fraction of positives.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax.plot(prob_pred, prob_true, "s-", color="#2a9d8f", linewidth=2, label="Calibrated QCSI")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ablation_curves(
    ablation_scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots ROC and PR curves for multiple ablation systems.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    colors = {
        "A0": "#6c757d", "A1": "#17a2b8", "A2": "#ffc107", "A3": "#fd7e14",
        "A4": "#20c997", "A5": "#6f42c1", "A6": "#007bff", "A7": "#dc3545"
    }

    # ROC Curves
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    for sys_id, scores in ablation_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        r_auc = auc(fpr, tpr)
        col = colors.get(sys_id, "#333333")
        ax1.plot(fpr, tpr, label=f"{sys_id} (AUC = {r_auc:.3f})", color=col, linewidth=1.8)

    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9)

    # PR Curves
    for sys_id, scores in ablation_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        p_auc = auc(rec, prec)
        col = colors.get(sys_id, "#333333")
        ax2.plot(rec, prec, label=f"{sys_id} (PR-AUC = {p_auc:.3f})", color=col, linewidth=1.8)

    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_comprehensive_dashboard(
    img_A: np.ndarray,
    img_B: np.ndarray,
    local_result: Dict[str, Any],
    global_result: Dict[str, Any],
    explanation: Dict[str, Any],
    save_path: str
):
    """
    Renders an informative 6-panel visual dashboard matching the architecture specification:
    - Panel 1: Image A with RootSIFT Keypoints
    - Panel 2: Image B with Keypoints
    - Panel 3: Inlier Correspondences with Color-Coded Quantum Fidelity
    - Panel 4: Density Matrix Eigenvalue Spectra (lambda_k)
    - Panel 5: Multimodal Component Score Breakdown (Bar Chart)
    - Panel 6: Quantum Information Card & Final Similarity Verdict
    """
    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.suptitle(
        f"QCSI Hybrid Quantum-Classical Verification Report: {explanation['calibrated_similarity_pct']:.1f}% Calibrated Similarity",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    feat_A = local_result["feat_A"]
    feat_B = local_result["feat_B"]
    geo = local_result.get("geometry", {})
    inlier_mask = geo.get("inlier_mask", np.array([]))
    matched = local_result.get("matched", {})
    pts_A = matched.get("pts_A", np.empty((0, 2)))
    pts_B = matched.get("pts_B", np.empty((0, 2)))
    fidelities = matched.get("fidelities", np.empty((0,)))

    # Panel 1: Image A Keypoints
    ax1 = plt.subplot(2, 3, 1)
    kp_draw_A = cv2.drawKeypoints(
        img_A, feat_A["keypoints"][:80], None, color=(0, 255, 100),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    ax1.imshow(kp_draw_A)
    ax1.set_title(f"Image A: {feat_A['num_kps']} Keypoints (S = {feat_A['entropy']:.2f} bits)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Image B Keypoints
    ax2 = plt.subplot(2, 3, 2)
    kp_draw_B = cv2.drawKeypoints(
        img_B, feat_B["keypoints"][:80], None, color=(255, 120, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    ax2.imshow(kp_draw_B)
    ax2.set_title(f"Image B: {feat_B['num_kps']} Keypoints (S = {feat_B['entropy']:.2f} bits)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Inlier Lines
    ax3 = plt.subplot(2, 3, 3)
    h_A, w_A = img_A.shape[:2]
    h_B, w_B = img_B.shape[:2]
    canvas = np.zeros((max(h_A, h_B), w_A + w_B, 3), dtype=np.uint8)
    canvas[:h_A, :w_A] = img_A
    canvas[:h_B, w_A:w_A + w_B] = img_B
    ax3.imshow(canvas)
    ax3.axis("off")
    ax3.set_title(f"Geometric Inliers: {geo.get('inlier_count', 0)} Matches Verified", fontsize=11, fontweight="bold")

    if len(pts_A) > 0 and len(inlier_mask) == len(pts_A):
        cmap = plt.get_cmap("viridis")
        for idx in range(len(pts_A)):
            if inlier_mask[idx]:
                pA = pts_A[idx]
                pB = pts_B[idx]
                x1, y1 = float(pA[0]), float(pA[1])
                x2, y2 = float(pB[0]) + w_A, float(pB[1])
                fid = fidelities[idx] if idx < len(fidelities) else 0.5
                line_color = cmap(fid)
                ax3.plot([x1, x2], [y1, y2], color=line_color, linewidth=1.5, alpha=0.85)

    # Panel 4: Eigenvalue Spectra
    ax4 = plt.subplot(2, 3, 4)
    eig_A = np.sort(np.linalg.eigvalsh(feat_A["density_matrix"]))[::-1][:25]
    eig_B = np.sort(np.linalg.eigvalsh(feat_B["density_matrix"]))[::-1][:25]
    ax4.plot(range(1, len(eig_A) + 1), eig_A, "o-", color="#1f77b4", label="Image A rho_A", linewidth=2)
    ax4.plot(range(1, len(eig_B) + 1), eig_B, "s--", color="#ff7f0e", label="Image B rho_B", linewidth=2)
    ax4.set_xlabel("Eigenvalue Rank k (7-Qubit Space)", fontsize=10)
    ax4.set_ylabel("Probability lambda_k", fontsize=10)
    ax4.set_title("Density Matrix Spectra & Purity", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=9)

    # Panel 5: Multimodal Component Scores
    ax5 = plt.subplot(2, 3, 5)
    comps = explanation["components"]
    comp_names = ["Geometric", "Semantic", "Quantum State"]
    comp_values = [comps.get("geometric_score", 0), comps.get("semantic_score", 0), comps.get("quantum_state_score", 0)]
    bar_colors = ["#2b5c8f", "#2a9d8f", "#e76f51"]
    bars = ax5.bar(comp_names, comp_values, color=bar_colors, edgecolor="black", width=0.5)
    ax5.set_ylim(0, 115)
    ax5.set_ylabel("Component Score (%)", fontsize=10)
    ax5.set_title("Component Score Breakdown", fontsize=11, fontweight="bold")
    for bar in bars:
        y = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, y + 2, f"{y:.1f}%", ha="center", va="bottom", fontsize=10)

    # Panel 6: Summary & Evidence Card
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    ev = explanation["evidence"]
    card_text = (
        f"QUANTUM & CLASSICAL EVIDENCE SUMMARY:\n"
        f"--------------------------------------------------\n"
        f"• Final Calibrated Similarity : {explanation['calibrated_similarity_pct']:.2f}%\n"
        f"• Verdict                     : {explanation['verdict']}\n"
        f"• RANSAC Inlier Count         : {ev.get('inlier_count', 0)}\n"
        f"• Inlier Ratio                : {ev.get('inlier_ratio', 0.0)*100:.2f}%\n"
        f"• Mean Inlier Fidelity K      : {ev.get('mean_inlier_fidelity', 0.0)*100:.2f}%\n"
        f"• Uhlmann Mixed Fidelity F    : {ev.get('uhlmann_fidelity', 0.0)*100:.2f}%\n"
        f"• Quantum JSD Divergence      : {ev.get('qjsd', 0.0):.6f}\n"
        f"• DINOv2 Cosine Similarity    : {ev.get('dinov2_cosine', 0.0):.4f}\n"
        f"• ZZFeatureMap Global Fidelity: {ev.get('quantum_global_fidelity', 0.0)*100:.2f}%\n"
        f"--------------------------------------------------\n"
        f"REASONING:\n{explanation['reasoning']}"
    )
    ax6.text(
        0.0, 0.98, card_text,
        transform=ax6.transAxes,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#2b5c8f", linewidth=1.5)
    )

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.96, hspace=0.35, wspace=0.25)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
