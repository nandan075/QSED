"""
Reversible Parallel Adder (PA) Module for Quantum Image Edge Detection (QSED).

Computes the quantum sum |A + B> of two n-qubit registers |A> and |B>.

Paper Reference:
- Section 3.1 (3): Reversible parallel adder (Figure 6, Ref. [33] Islam et al.).
"""

from qiskit import QuantumCircuit, QuantumRegister


def build_parallel_adder(n_bits: int) -> QuantumCircuit:
    """
    Construct Reversible Parallel Adder (PA) quantum circuit.

    Purpose:
        Addition of two binary registers |A> and |B>. Used in:
        1. Sobel mask linear sum evaluation (Step 3).
        2. Subtraction via complement addition: A - B = A + (~B + 1) (Equation 7).

    Inputs:
        n_bits (int): Bit width of registers A and B.

    Outputs:
        QuantumCircuit: Qiskit gate block implementing PA logic.

    Time Complexity:
        O(n_bits) gates.

    Space Complexity:
        2*n_bits + 1 (carry) qubits.

    Reference:
        Paper Section 3.1 (3), Figure 6.
    """
    a_reg = QuantumRegister(n_bits, name='a')
    b_reg = QuantumRegister(n_bits, name='b')
    carry = QuantumRegister(1, name='c_in')

    qc = QuantumCircuit(a_reg, b_reg, carry, name=f'PA_{n_bits}bit')

    # Ripple-carry addition logic using CNOT and Toffoli (CCX) gates
    # Carry generation phase
    for i in range(n_bits - 1):
        qc.ccx(a_reg[i], b_reg[i], a_reg[i + 1])
        qc.cx(a_reg[i], b_reg[i])
        qc.ccx(b_reg[i], a_reg[i + 1], carry[0])

    # Final bit sum
    qc.cx(a_reg[n_bits - 1], b_reg[n_bits - 1])

    # Uncompute carry bits to maintain reversibility
    for i in reversed(range(n_bits - 1)):
        qc.ccx(b_reg[i], a_reg[i + 1], carry[0])
        qc.cx(a_reg[i], b_reg[i])
        qc.ccx(a_reg[i], b_reg[i], a_reg[i + 1])
        qc.cx(a_reg[i], b_reg[i])

    return qc
