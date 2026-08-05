"""
Novel Enhanced Quantum Representation (NEQR) Module for Quantum Image Edge Detection (QSED).

This module implements:
1. NEQR Quantum Circuit construction for 2^n x 2^n grayscale images (Equation 1).
2. Decoding/reconstruction of grayscale pixel matrices from Quantum Statevectors or Sampler results.
3. NEQR state preparation with auxiliary qubits for 5x5 neighborhood shift (Equation 8).

Paper Reference:
- Section 2.1: NEQR representation model (Equations 1 & 2, Figure 1).
- Section 3.2: Step 1 NEQR images preparation (Equation 8).
"""

from typing import Tuple, Optional, Dict
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


class NEQRImage:
    """
    NEQR Quantum Image Class representing a 2^n x 2^n grayscale image as an entangled quantum superposition.

    Equation (1):
        |I> = (1 / 2^n) * sum_{Y=0}^{2^n-1} sum_{X=0}^{2^n-1} |C_YX> |Y> |X>
        where |C_YX> = |C_{q-1} C_{q-2} ... C_0> is the q-qubit intensity register,
        |Y> = |Y_{n-1} ... Y_0> is the n-qubit Y-position register,
        |X> = |X_{n-1} ... X_0> is the n-qubit X-position register.
    """

    def __init__(self, image_matrix: np.ndarray, q_bits: int = 8):
        """
        Initialize NEQR image parameters and validate dimensions.

        Inputs:
            image_matrix (np.ndarray): 2D square image matrix of size 2^n x 2^n.
            q_bits (int): Bit-depth for grayscale intensities (default 8).
        """
        if image_matrix.ndim != 2:
            raise ValueError(f"Image matrix must be 2D, got shape {image_matrix.shape}")
        h, w = image_matrix.shape
        if h != w:
            raise ValueError(f"Image must be square 2^n x 2^n, got shape ({h}, {w})")
        if (h & (h - 1) != 0) or h < 1:
            raise ValueError(f"Image size must be a power of 2, got size {h}")

        self.h = h
        self.w = w
        self.n = int(np.log2(h))
        self.q_bits = q_bits
        self.image_matrix = image_matrix.astype(np.uint8)

        # Total qubits: q intensity qubits + n Y-position qubits + n X-position qubits = 2n + q
        self.num_qubits = 2 * self.n + self.q_bits

    def build_circuit(self) -> QuantumCircuit:
        """
        Build the Qiskit QuantumCircuit that constructs the NEQR superposition state |I>.

        Purpose:
            1. Apply Hadamard gates H to all Y and X position qubits to put the system in uniform spatial superposition.
            2. For each pixel (Y, X) with intensity C(Y,X), apply multi-controlled NOT (MCX) gates targeted on the
               grayscale register |C_YX> conditioned on position qubits |Y> |X>.

        Outputs:
            QuantumCircuit: Qiskit quantum circuit encoding the NEQR image.

        Time Complexity:
            O(2^{2n} * q) quantum gate operations.

        Space Complexity:
            O(2n + q) qubits.

        Reference:
            Paper Section 2.1, Equation (1), Equation (2), Figure 1.
        """
        q_reg = QuantumRegister(self.q_bits, name='color')
        y_reg = QuantumRegister(self.n, name='y_pos')
        x_reg = QuantumRegister(self.n, name='x_pos')
        qc = QuantumCircuit(q_reg, y_reg, x_reg, name='NEQR_Image')

        # Step 1: Put position registers |Y> and |X> into uniform superposition using Hadamard gates
        for i in range(self.n):
            qc.h(y_reg[i])
            qc.h(x_reg[i])

        # Step 2: Encode pixel grayscale values |C_YX> conditioned on position |Y>|X>
        for y in range(self.h):
            for x in range(self.w):
                intensity = int(self.image_matrix[y, x])
                if intensity == 0:
                    continue  # Color bits remain |0> by default

                # Binary representation of position y and x
                y_bin = format(y, f'0{self.n}b')
                x_bin = format(x, f'0{self.n}b')

                # Identify position qubits that are '0' in the binary index to flip them using X gates
                # so that control activates on state |y>|x>
                zeros_to_flip = []

                for idx, bit in enumerate(y_bin):
                    if bit == '0':
                        # y_reg[0] is MSB in string format, let's keep consistent indexing
                        # y_reg index from MSB to LSB
                        qubit = y_reg[self.n - 1 - idx]
                        zeros_to_flip.append(qubit)
                        qc.x(qubit)

                for idx, bit in enumerate(x_bin):
                    if bit == '0':
                        qubit = x_reg[self.n - 1 - idx]
                        zeros_to_flip.append(qubit)
                        qc.x(qubit)

                # All position qubits (y_reg + x_reg) form the control sequence
                controls = list(y_reg) + list(x_reg)

                # Binary representation of intensity C_YX
                c_bin = format(intensity, f'0{self.q_bits}b')

                for c_idx, c_bit in enumerate(c_bin):
                    if c_bit == '1':
                        target_qubit = q_reg[self.q_bits - 1 - c_idx]
                        if len(controls) == 1:
                            qc.cx(controls[0], target_qubit)
                        elif len(controls) == 2:
                            qc.ccx(controls[0], controls[1], target_qubit)
                        else:
                            qc.mcx(controls, target_qubit)

                # Unflip position qubits to restore position register
                for qubit in zeros_to_flip:
                    qc.x(qubit)

        return qc

    @staticmethod
    def decode_statevector(statevector: Statevector, n: int, q_bits: int = 8) -> np.ndarray:
        """
        Reconstruct 2^n x 2^n pixel intensity matrix directly from a quantum statevector.

        Inputs:
            statevector (Statevector): Qiskit Statevector of NEQR circuit.
            n (int): Number of position qubits per axis (image size 2^n x 2^n).
            q_bits (int): Bit-depth of intensity register.

        Outputs:
            np.ndarray: Reconstructed grayscale image of shape (2^n, 2^n).

        Reference:
            Paper Section 2.1, Equation (1).
        """
        side = 1 << n
        reconstructed = np.zeros((side, side), dtype=np.uint8)
        state_dict = statevector.to_dict()

        for bitstring, amplitude in state_dict.items():
            if np.abs(amplitude) > 1e-6:
                # Bitstring format in Qiskit is reverse of register order:
                # rightmost = color[0], leftmost = x_pos[n-1]
                # Qiskit qubit ordering: q_reg (q), y_reg (n), x_reg (n)
                # Bitstring layout: x_pos[n-1..0] y_pos[n-1..0] color[q-1..0]
                total_len = 2 * n + q_bits
                clean_bitstring = bitstring.zfill(total_len)

                x_str = clean_bitstring[:n]
                y_str = clean_bitstring[n:2*n]
                c_str = clean_bitstring[2*n:]

                x = int(x_str, 2)
                y = int(y_str, 2)
                c = int(c_str, 2)

                if 0 <= y < side and 0 <= x < side:
                    reconstructed[y, x] = c

        return reconstructed


def prepare_neqr_with_auxiliary(
    image_matrix: np.ndarray,
    q_bits: int = 8
) -> QuantumCircuit:
    """
    Prepare NEQR quantum image state tensor-producted with 24 * q auxiliary qubits per Step 1, Equation (8).

    Equation (8):
        |0>^{\\otimes 24q} \\otimes |I> = (1 / 2^n) \\sum_{Y=0}^{2^n-1} \\sum_{X=0}^{2^n-1} |0>^{\\otimes 24q} |C_{YX}> |Y> |X>

    Inputs:
        image_matrix (np.ndarray): 2^n x 2^n grayscale matrix.
        q_bits (int): Grayscale intensity bit depth.

    Outputs:
        QuantumCircuit: Quantum circuit containing 24q auxiliary qubits + NEQR registers.

    Reference:
        Paper Section 3.2, Step 1, Equation (8).
    """
    neqr_img = NEQRImage(image_matrix, q_bits=q_bits)
    base_qc = neqr_img.build_circuit()

    aux_reg = QuantumRegister(24 * q_bits, name='aux_shift_24q')
    full_qc = QuantumCircuit(aux_reg, *base_qc.qregs, name='NEQR_Prepared_Step1')

    # Compose base circuit into full circuit
    full_qc.compose(base_qc, qubits=list(full_qc.qubits)[24*q_bits:], inplace=True)
    return full_qc
