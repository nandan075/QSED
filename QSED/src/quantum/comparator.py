"""
Quantum Comparator (QC) Module for Quantum Image Edge Detection (QSED).

Compares two n-qubit bitstrings |A> = |a_{n-1} ... a_0> and |B> = |b_{n-1} ... b_0>.
Outputs two flag bits (C_1, C_0):
- If A > B: C_1 = 1, C_0 = 0
- If A < B: C_1 = 0, C_0 = 1
- If A = B: C_1 = 0, C_0 = 0

Paper Reference:
- Section 3.1 (1): Quantum comparator (Figure 4, Ref. [30] Oliveira & Ramos).
"""

from qiskit import QuantumCircuit, QuantumRegister


def build_quantum_comparator(n_bits: int) -> QuantumCircuit:
    """
    Construct Quantum Comparator (QC) circuit for two n-qubit registers |A> and |B>.

    Purpose:
        Compare binary numbers bit-by-bit from MSB to LSB. Used extensively in:
        1. Max gradient selection tree (Step 3).
        2. Non-maximum suppression (Step 4, Fig. 17).
        3. Double threshold classification (Step 5, Fig. 18).
        4. Edge tracking 24-neighborhood check (Step 6, Fig. 19).

    Inputs:
        n_bits (int): Number of qubits in registers A and B.

    Outputs:
        QuantumCircuit: Qiskit gate block implementing QC logic.

    Truth Table:
        Condition   | C_1 | C_0
        -----------------------
        A > B       |  1  |  0
        A < B       |  0  |  1
        A == B      |  0  |  0

    Time Complexity:
        O(n_bits) logic gate operations.

    Space Complexity:
        2*n_bits input qubits + 2 output flag qubits (C_1, C_0) + n_bits auxiliary qubits.

    Reference:
        Paper Section 3.1 (1), Figure 4.
    """
    a_reg = QuantumRegister(n_bits, name='a')
    b_reg = QuantumRegister(n_bits, name='b')
    c_out = QuantumRegister(2, name='c_out')  # c_out[1] = C_1 (A>B), c_out[0] = C_0 (A<B)
    aux = QuantumRegister(n_bits, name='aux')

    qc = QuantumCircuit(a_reg, b_reg, c_out, aux, name=f'QC_{n_bits}bit')

    # Bit-wise equality e_k = NOT(a_k XOR b_k) using CNOT and X gates
    # e_k = 1 if a_k == b_k, 0 otherwise
    for k in range(n_bits):
        qc.cx(a_reg[k], aux[k])
        qc.cx(b_reg[k], aux[k])
        qc.x(aux[k])

    # MSB to LSB comparison logic:
    # A > B if at MSB k where a_k != b_k, a_k = 1 and b_k = 0
    # A < B if at MSB k where a_k != b_k, a_k = 0 and b_k = 1

    # Check MSB down to LSB
    for k in reversed(range(n_bits)):
        # Higher bits equality prefix: e_{n-1} ... e_{k+1}
        eq_controls = [aux[i] for i in range(k + 1, n_bits)]

        # C_1 (A > B): Activated if higher bits equal, a_k = 1, b_k = 0
        # Condition: eq_controls + a_k + NOT(b_k)
        qc.x(b_reg[k])
        c1_controls = eq_controls + [a_reg[k], b_reg[k]]
        if len(c1_controls) == 1:
            qc.cx(c1_controls[0], c_out[1])
        elif len(c1_controls) == 2:
            qc.ccx(c1_controls[0], c1_controls[1], c_out[1])
        else:
            qc.mcx(c1_controls, c_out[1])
        qc.x(b_reg[k])  # Restore b_reg[k]

        # C_0 (A < B): Activated if higher bits equal, a_k = 0, b_k = 1
        # Condition: eq_controls + NOT(a_k) + b_k
        qc.x(a_reg[k])
        c0_controls = eq_controls + [a_reg[k], b_reg[k]]
        if len(c0_controls) == 1:
            qc.cx(c0_controls[0], c_out[0])
        elif len(c0_controls) == 2:
            qc.ccx(c0_controls[0], c0_controls[1], c_out[0])
        else:
            qc.mcx(c0_controls, c_out[0])
        qc.x(a_reg[k])  # Restore a_reg[k]

    return qc
