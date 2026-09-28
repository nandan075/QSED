"""
Unit tests for QCSI preprocessing pipeline.
"""

import numpy as np
import pytest
from PIL import Image
from qcsi.preprocess import ImagePreprocessor


def test_image_preprocessor_loading():
    preproc = ImagePreprocessor(max_dim=512)

    # Test array loading
    dummy_arr = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    loaded = preproc.load_image(dummy_arr)
    assert loaded.shape == (200, 300, 3)

    # Test PIL loading
    pil_img = Image.fromarray(dummy_arr)
    loaded_pil = preproc.load_image(pil_img)
    assert loaded_pil.shape == (200, 300, 3)


def test_resize_keep_aspect():
    preproc = ImagePreprocessor(max_dim=500)
    large_img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    resized = preproc.resize_keep_aspect(large_img, max_dim=500)
    assert max(resized.shape[:2]) <= 500
    assert resized.shape[1] == 500
    assert resized.shape[0] == 250


def test_clahe_and_dual_preproc():
    preproc = ImagePreprocessor()
    img_rgb = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)

    gray, clahe_gray = preproc.prepare_local(img_rgb)
    assert gray.shape == (300, 300)
    assert clahe_gray.shape == (300, 300)

    dino_rgb = preproc.prepare_global(img_rgb)
    assert dino_rgb.shape == (224, 224, 3)


def test_preprocess_pair():
    preproc = ImagePreprocessor(max_dim=400)
    img_A = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img_B = np.random.randint(0, 255, (350, 350, 3), dtype=np.uint8)

    pair = preproc.preprocess_pair(img_A, img_B)
    assert "gray_A" in pair
    assert "gray_B" in pair
    assert "dino_A" in pair
    assert "dino_B" in pair
