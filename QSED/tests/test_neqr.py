"""
Unit Tests for NEQR Quantum Image Representation.

Tests:
1. Circuit construction for 2x2 and 4x4 images.
2. Statevector decoding roundtrip matching original pixel values (Equation 1, Equation 2).
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from src.quantum.neqr import NEQRImage, prepare_neqr_with_auxiliary


def test_neqr_2x2_encoding():
    """
    Test NEQR encoding and decoding on 2x2 example from Paper Section 2.1 Equation (2).
    Image: [[0, 100], [200, 255]]
    """
    img_2x2 = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    neqr = NEQRImage(img_2x2, q_bits=8)
    qc = neqr.build_circuit()

    # Verify qubit count: 2*n + q = 2*1 + 8 = 10 qubits
    assert qc.num_qubits == 10

    sv = Statevector.from_instruction(qc)
    reconstructed = NEQRImage.decode_statevector(sv, n=1, q_bits=8)

    np.testing.assert_array_equal(reconstructed, img_2x2)


def test_neqr_4x4_encoding():
    """
    Test NEQR encoding and statevector reconstruction on a 4x4 synthetic image.
    """
    img_4x4 = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
        [130, 140, 150, 160]
    ], dtype=np.uint8)

    neqr = NEQRImage(img_4x4, q_bits=8)
    qc = neqr.build_circuit()

    # Qubit count: 2*n + q = 2*2 + 8 = 12 qubits
    assert qc.num_qubits == 12

    sv = Statevector.from_instruction(qc)
    reconstructed = NEQRImage.decode_statevector(sv, n=2, q_bits=8)

    np.testing.assert_array_equal(reconstructed, img_4x4)


def test_neqr_prepared_with_auxiliary():
    """
    Test NEQR state preparation with 24q auxiliary qubits per Step 1 Equation (8).
    """
    img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    full_qc = prepare_neqr_with_auxiliary(img, q_bits=8)

    # Total qubits: 24*8 aux + 2*1 pos + 8 col = 192 + 10 = 202 qubits
    assert full_qc.num_qubits == 202
