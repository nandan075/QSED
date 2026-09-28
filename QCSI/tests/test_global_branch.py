"""
Unit tests for QCSI Branch B (Global Semantic Matcher).
"""

import numpy as np
import pytest
from qcsi.global_branch import DINOv2Extractor, GlobalSemanticMatcher


def test_dinov2_extractor():
    extractor = DINOv2Extractor(use_fallback_if_offline=True)
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    emb = extractor.extract_embedding(dummy_img)
    assert emb.shape == (768,)
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 1e-5) == 1.0


def test_global_semantic_matcher():
    matcher = GlobalSemanticMatcher(n_qubits=8)
    img_A = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_B = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    res = matcher.compute_global_similarity(img_A, img_B)
    assert -1.0 <= res["classical_cosine"] <= 1.0
    assert 0.0 <= res["quantum_fidelity"] <= 1.0
    assert len(res["angles_A"]) == 8
    assert len(res["angles_B"]) == 8


def test_global_self_fidelity():
    matcher = GlobalSemanticMatcher(n_qubits=8)
    img_A = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    res = matcher.compute_global_similarity(img_A, img_A)
    assert pytest.approx(res["classical_cosine"], 1e-4) == 1.0
    assert pytest.approx(res["quantum_fidelity"], 1e-4) == 1.0
