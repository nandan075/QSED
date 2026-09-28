"""
QCSI Gallery Search Mode: Two-Stage Re-Ranking Engine.

Stage 1: High-speed shortlist using DINOv2 768-d embeddings in a FAISS index (Top-K, default 50).
Stage 2: Precision geometric verification & quantum calibrated fusion re-ranking.
"""

from typing import List, Dict, Any, Union, Optional, Tuple
import os
import numpy as np
import faiss

from qcsi.preprocess import ImagePreprocessor
from qcsi.local_branch import LocalGeometryMatcher
from qcsi.global_branch import GlobalSemanticMatcher
from qcsi.fusion import CalibratedFusionModel, extract_feature_vector, explain_similarity


class GallerySearchEngine:
    """Scalable image search over gallery databases with quantum re-ranking."""

    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        local_matcher: Optional[LocalGeometryMatcher] = None,
        global_matcher: Optional[GlobalSemanticMatcher] = None,
        fusion_model: Optional[CalibratedFusionModel] = None,
    ):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.local_matcher = local_matcher or LocalGeometryMatcher()
        self.global_matcher = global_matcher or GlobalSemanticMatcher()
        self.fusion_model = fusion_model or CalibratedFusionModel()

        self.index = faiss.IndexFlatIP(768)
        self.image_records: List[Dict[str, Any]] = []

    def index_image(self, image_id: str, image_source: Union[str, np.ndarray]) -> int:
        """
        Adds a single image to the gallery index.
        """
        img_rgb = self.preprocessor.load_image(image_source)
        emb = self.global_matcher.extractor.extract_embedding(img_rgb)
        
        # FAISS expects float32
        emb_f32 = emb.astype(np.float32).reshape(1, -1)
        self.index.add(emb_f32)

        idx = len(self.image_records)
        record = {
            "id": image_id,
            "index_pos": idx,
            "source": image_source if isinstance(image_source, str) else None,
            "img_rgb": img_rgb if not isinstance(image_source, str) else None,
        }
        self.image_records.append(record)
        return idx

    def index_batch(self, image_items: List[Tuple[str, Union[str, np.ndarray]]]):
        """Indexes multiple images in batch."""
        for item_id, item_src in image_items:
            self.index_image(item_id, item_src)

    def search(
        self,
        query_image: Union[str, np.ndarray],
        top_k: int = 50,
        rerank_limit: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
        1. Fast DINOv2 FAISS shortlist (top_k).
        2. Quantum geometric Branch A + Calibrated Fusion re-ranking on the top candidates.
        """
        if self.index.ntotal == 0:
            return []

        # Preprocess query
        query_rgb = self.preprocessor.load_image(query_image)
        query_gray, _ = self.preprocessor.prepare_local(query_rgb)
        query_emb = self.global_matcher.extractor.extract_embedding(query_rgb).astype(np.float32).reshape(1, -1)

        # Stage 1: Shortlist retrieval
        effective_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, effective_k)

        candidates = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.image_records):
                continue
            candidates.append({
                "gallery_id": self.image_records[idx]["id"],
                "index_pos": int(idx),
                "stage1_dinov2_cosine": float(score),
                "initial_rank": rank + 1,
            })

        # Stage 2: Re-ranking top candidates with Branch A & Quantum Fusion
        num_to_rerank = min(len(candidates), rerank_limit)
        results = []

        for i in range(num_to_rerank):
            cand = candidates[i]
            rec = self.image_records[cand["index_pos"]]
            cand_src = rec["source"] if rec["source"] is not None else rec["img_rgb"]
            cand_rgb = self.preprocessor.load_image(cand_src)
            cand_gray, _ = self.preprocessor.prepare_local(cand_rgb)

            # Local Branch A
            local_res = self.local_matcher.compute_local_similarity(query_gray, cand_gray)

            # Global Branch B
            global_res = self.global_matcher.compute_global_similarity(query_rgb, cand_rgb)

            # Calibrated Fusion
            features = extract_feature_vector(local_res, global_res)
            calibrated_prob = self.fusion_model.predict_probability(features)
            explanation = explain_similarity(local_res, global_res, calibrated_prob)

            cand_result = {
                "gallery_id": cand["gallery_id"],
                "initial_rank": cand["initial_rank"],
                "stage1_dinov2_cosine": cand["stage1_dinov2_cosine"],
                "calibrated_similarity_pct": explanation["calibrated_similarity_pct"],
                "calibrated_probability": calibrated_prob,
                "verdict": explanation["verdict"],
                "reasoning": explanation["reasoning"],
                "evidence": explanation["evidence"],
                "components": explanation["components"],
            }
            results.append(cand_result)

        # Append remaining candidates without expensive re-ranking
        for i in range(num_to_rerank, len(candidates)):
            cand = candidates[i]
            results.append({
                "gallery_id": cand["gallery_id"],
                "initial_rank": cand["initial_rank"],
                "stage1_dinov2_cosine": cand["stage1_dinov2_cosine"],
                "calibrated_similarity_pct": round(max(0.0, cand["stage1_dinov2_cosine"]) * 100.0, 2),
                "calibrated_probability": max(0.0, cand["stage1_dinov2_cosine"]),
                "verdict": "Candidate (Unverified)",
                "reasoning": "Retrieved in top-K shortlist via global DINOv2 embedding.",
                "evidence": {"dinov2_cosine": cand["stage1_dinov2_cosine"]},
                "components": {},
            })

        # Sort by calibrated probability descending
        results.sort(key=lambda r: r["calibrated_probability"], reverse=True)
        for r_idx, item in enumerate(results):
            item["final_rank"] = r_idx + 1

        return results
