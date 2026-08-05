"""
Integration Tests for Complete QSED Pipeline.

Tests end-to-end execution of NEQR preparation, shift, gradient, NMS, thresholding, and hysteresis tracking.

Paper Reference: Section 3.2, Section 4.2.
"""

import numpy as np
import pytest
from src.quantum.qsed import QSEDRunner
from src.classical.image_loader import generate_synthetic_image
from src.utils.metrics import calculate_mse


def test_qsed_pipeline_synthetic_checkerboard():
    """
    Test end-to-end QSED runner on 16x16 synthetic checkerboard image.
    """
    img = generate_synthetic_image(size=16, pattern='checkerboard')

    runner = QSEDRunner(q_bits=8, mode='hybrid')
    results = runner.run_pipeline(img)

    assert 'original' in results
    assert 'gradient_magnitude' in results
    assert 'nms_image' in results
    assert 'threshold_map' in results
    assert 'final_edges' in results

    final_edges = results['final_edges']
    assert final_edges.shape == (16, 16)
    # Edge map should contain non-zero edge pixels
    assert np.sum(final_edges) > 0


def test_qsed_quantum_circuit_mode_2x2():
    """
    Test QSED pipeline in 'quantum' circuit execution mode for 2x2 image.
    """
    img_2x2 = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    runner = QSEDRunner(q_bits=8, mode='quantum')
    results = runner.run_pipeline(img_2x2)

    assert 'neqr_circuit' in results
    assert results['final_edges'].shape == (2, 2)
