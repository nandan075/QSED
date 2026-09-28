"""
QCSI FastAPI Service: Pairwise Image Comparison and Gallery Search.

Endpoints:
- POST /compare: Upload two images to get calibrated similarity percentage and evidence.
- POST /gallery/index: Index images into the FAISS gallery database.
- POST /gallery/search: Retrieve and quantum re-rank candidate images.
- GET /health: Health check, active hardware, and loaded components.
"""

from typing import Optional, List
import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from qcsi.preprocess import ImagePreprocessor
from qcsi.local_branch import LocalGeometryMatcher
from qcsi.global_branch import GlobalSemanticMatcher
from qcsi.fusion import CalibratedFusionModel, extract_feature_vector, explain_similarity
from qcsi.gallery import GallerySearchEngine

app = FastAPI(
    title="QCSI: Hybrid Quantum-Classical Image Similarity API",
    description="High-precision image verification combining 7-qubit amplitude-encoded local geometry, frozen DINOv2 global semantics, and parameterized quantum feature maps.",
    version="1.0.0",
)

# Global singleton pipelines
preprocessor = ImagePreprocessor()
local_matcher = LocalGeometryMatcher()
global_matcher = GlobalSemanticMatcher()
fusion_model = CalibratedFusionModel()
gallery_engine = GallerySearchEngine(
    preprocessor=preprocessor,
    local_matcher=local_matcher,
    global_matcher=global_matcher,
    fusion_model=fusion_model,
)


def render_match_visualization(
    img_A: np.ndarray,
    img_B: np.ndarray,
    local_res: dict,
    calibrated_pct: float
) -> str:
    """Renders a base64-encoded visual plot of geometric inlier correspondences."""
    geo = local_res.get("geometry", {})
    inlier_mask = geo.get("inlier_mask", np.array([]))
    matched = local_res.get("matched", {})
    pts_A = matched.get("pts_A", np.empty((0, 2)))
    pts_B = matched.get("pts_B", np.empty((0, 2)))

    h_A, w_A = img_A.shape[:2]
    h_B, w_B = img_B.shape[:2]
    canvas_h = max(h_A, h_B)
    canvas_w = w_A + w_B

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_A, :w_A] = img_A
    canvas[:h_B, w_A:w_A + w_B] = img_B

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(
        f"QCSI Inlier Correspondence: Calibrated Similarity = {calibrated_pct:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    if len(pts_A) > 0 and len(inlier_mask) == len(pts_A):
        for idx in range(len(pts_A)):
            pA = pts_A[idx]
            pB = pts_B[idx]
            x1, y1 = float(pA[0]), float(pA[1])
            x2, y2 = float(pB[0]) + w_A, float(pB[1])

            if inlier_mask[idx]:
                ax.plot([x1, x2], [y1, y2], color="#00ff66", linewidth=1.2, alpha=0.85)
                ax.plot(x1, y1, "o", color="#00ffff", markersize=3)
                ax.plot(x2, y2, "o", color="#ff00ff", markersize=3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


@app.get("/health")
def health_check():
    """System health check and hardware acceleration info."""
    import torch
    import qiskit
    return {
        "status": "online",
        "service": "QCSI Hybrid Quantum-Classical Image Similarity",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device": global_matcher.extractor.device,
        "qiskit_version": qiskit.__version__,
        "gallery_size": gallery_engine.index.ntotal,
    }


class ComparePathRequest(BaseModel):
    image_path_A: str
    image_path_B: str
    include_visualization: bool = False


@app.post("/compare/paths")
def compare_image_paths(req: ComparePathRequest):
    """Pairwise image comparison by filesystem paths."""
    try:
        prep = preprocessor.preprocess_pair(req.image_path_A, req.image_path_B)
        local_res = local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])
        global_res = global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])

        feat_vec = extract_feature_vector(local_res, global_res)
        calibrated_prob = fusion_model.predict_probability(feat_vec)
        explanation = explain_similarity(local_res, global_res, calibrated_prob)

        resp = {
            "image_A": req.image_path_A,
            "image_B": req.image_path_B,
            **explanation,
        }

        if req.include_visualization:
            resp["visualization_base64"] = render_match_visualization(
                prep["rgb_A"], prep["rgb_B"], local_res, explanation["calibrated_similarity_pct"]
            )

        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_images(
    file_A: UploadFile = File(...),
    file_B: UploadFile = File(...),
    include_visualization: bool = Form(False)
):
    """Pairwise image comparison via multipart file upload."""
    try:
        bytes_A = await file_A.read()
        bytes_B = await file_B.read()

        prep = preprocessor.preprocess_pair(bytes_A, bytes_B)
        local_res = local_matcher.compute_local_similarity(prep["gray_A"], prep["gray_B"])
        global_res = global_matcher.compute_global_similarity(prep["rgb_A"], prep["rgb_B"])

        feat_vec = extract_feature_vector(local_res, global_res)
        calibrated_prob = fusion_model.predict_probability(feat_vec)
        explanation = explain_similarity(local_res, global_res, calibrated_prob)

        resp = {
            "filename_A": file_A.filename,
            "filename_B": file_B.filename,
            **explanation,
        }

        if include_visualization:
            resp["visualization_base64"] = render_match_visualization(
                prep["rgb_A"], prep["rgb_B"], local_res, explanation["calibrated_similarity_pct"]
            )

        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gallery/index")
async def index_gallery_image(
    image_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Uploads and indexes an image into the FAISS gallery."""
    try:
        data = await file.read()
        idx = gallery_engine.index_image(image_id, data)
        return {
            "status": "indexed",
            "image_id": image_id,
            "index_position": idx,
            "total_gallery_images": gallery_engine.index.ntotal,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gallery/search")
async def search_gallery(
    query_file: UploadFile = File(...),
    top_k: int = Form(50),
    rerank_limit: int = Form(10)
):
    """Queries the gallery with an uploaded image and performs quantum re-ranking."""
    try:
        data = await query_file.read()
        results = gallery_engine.search(data, top_k=top_k, rerank_limit=rerank_limit)
        return {
            "query_file": query_file.filename,
            "matches_count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
