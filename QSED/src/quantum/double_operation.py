"""
Quantum Double Operation (DO) Module for Quantum Image Edge Detection (QSED).

Multiplies a binary integer register by 2 (left bit-shift) using SWAP gates and auxiliary qubits.

Paper Reference:
- Section 3.1 (5): Quantum double operation (Figure 9, Ref. [28, 38] Chetia et al., Li & Liu).
"""

from qiskit import QuantumCircuit, QuantumRegister


def build_double_operation(n_bits: int) -> QuantumCircuit:
    """
    Construct Quantum Double Operation (DO) circuit.

    Purpose:
        Multiply binary integer register by 2 (2 * X) by shifting all bit positions to the left by 1 position.
        In 8-direction Sobel masks, coefficients 2 and 4 are frequent (e.g. 2*p, 4*p).
        Coefficients of 2 are computed using DO, and 4 using DO applied twice.

    Inputs:
        n_bits (int): Width of binary register.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing DO operation.

    Time Complexity:
        O(n_bits) SWAP gate operations.

    Space Complexity:
        n_bits input qubits + 1 auxiliary LSB qubit (initialized to |0>).

    Reference:
        Paper Section 3.1 (5), Figure 9.
    """
    in_reg = QuantumRegister(n_bits, name='in_val')
    out_reg = QuantumRegister(n_bits + 1, name='out_doubled')  # Extra MSB qubit for overflow

    qc = QuantumCircuit(in_reg, out_reg, name=f'DO_{n_bits}bit')

    # Shift bits: out_reg[i+1] <= in_reg[i], out_reg[0] <= 0
    # Copy in_reg[i] to out_reg[i+1] using CNOT gates
    for i in range(n_bits):
        qc.cx(in_reg[i], out_reg[i + 1])

    # LSB out_reg[0] remains |0> (representing multiplication by 2 in binary base)
    return qc
