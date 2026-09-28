"""
Quantum Absolute Value Operation (AV) Module for Quantum Image Edge Detection (QSED).

Calculates the absolute difference |A - B| between two integer registers |A> and |B>.

Paper Reference:
- Section 3.1 (4): Quantum absolute value operation (Equation 7, Figure 8).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .parallel_adder import build_parallel_adder
from .complement import build_complement_operation


def build_absolute_value(n_bits: int) -> QuantumCircuit:
    """
    Construct Quantum Absolute Value Operation (AV) circuit for |A - B|.

    Purpose:
        Compute |A - B| in quantum circuit. Required for computing directional gradient magnitudes
        |G_0|, |G_22.5|, ..., |G_157.5| per Equation (10) and Equation (11).

    Equation (7):
        A - B = A + (-B) = A + ([B]_{CA} + 1)
        If sign bit d_n == 1 (result negative), apply CA again to obtain magnitude.

    Inputs:
        n_bits (int): Bit width of input registers A and B.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing AV operation.

    Time Complexity:
        O(n_bits^2) or O(n_bits) depending on adder depth.

    Space Complexity:
        2*n_bits input + (n_bits + 1) output qubits + aux qubits.

    Reference:
        Paper Section 3.1 (4), Equation (7), Figure 8.
    """
    a_reg = QuantumRegister(n_bits, name='a')
    b_reg = QuantumRegister(n_bits, name='b')
    diff_reg = QuantumRegister(n_bits + 1, name='diff_signed')  # n bits value + 1 sign bit
    aux = QuantumRegister(1, name='carry_aux')

    qc = QuantumCircuit(a_reg, b_reg, diff_reg, aux, name=f'AV_{n_bits}bit')

    # Step 1: Compute complement of B: [B]_{CA}
    ca_b = build_complement_operation(n_bits)
    # Apply bitwise complement logic to B
    for i in range(n_bits):
        qc.x(b_reg[i])

    # Step 2: Add A and complement of B using Parallel Adder
    pa_gate = build_parallel_adder(n_bits)
    qc.append(pa_gate, list(a_reg) + list(b_reg) + list(aux))

    # Step 3: Copy result into signed difference register diff_reg
    for i in range(n_bits):
        qc.cx(b_reg[i], diff_reg[i])
    qc.cx(aux[0], diff_reg[n_bits])  # Sign bit d_n

    # Step 4: If sign bit diff_reg[n_bits] == 1, apply CA to get absolute magnitude
    ca_final = build_complement_operation(n_bits)
    qc.append(ca_final, list(diff_reg[:n_bits]) + [diff_reg[n_bits]])

    return qc
