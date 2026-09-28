"""
Unit tests for QCSI Gallery Search and Re-Ranking Engine.
"""

import numpy as np
import cv2
import pytest
from qcsi.gallery import GallerySearchEngine


def create_gallery_sample(text: str):
    img = np.zeros((200, 200, 3), dtype=np.uint8) + 200
    cv2.putText(img, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def test_gallery_search():
    engine = GallerySearchEngine()

    img1 = create_gallery_sample("IMAGE_1")
    img2 = create_gallery_sample("IMAGE_2")
    img3 = create_gallery_sample("IMAGE_3")

    engine.index_image("img_1", img1)
    engine.index_image("img_2", img2)
    engine.index_image("img_3", img3)

    assert engine.index.ntotal == 3

    # Query with identical image 1
    results = engine.search(img1, top_k=3, rerank_limit=3)
    assert len(results) == 3
    assert results[0]["gallery_id"] == "img_1"
    assert results[0]["calibrated_probability"] >= results[1]["calibrated_probability"]
