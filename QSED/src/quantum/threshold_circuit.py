"""
Double Threshold Detection Quantum Circuit Module for QSED.

Designs the quantum circuit for double threshold classification per Equation (14) and Figure 18.

Paper Reference:
- Section 3.2: Step 5 Double threshold detection (Equation 14, Figure 18).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .comparator import build_quantum_comparator


def build_double_threshold_circuit(q_bits: int = 8) -> QuantumCircuit:
    """
    Construct Double Threshold Detection quantum circuit.

    Purpose:
        Classify pixel gradient magnitude |G> into three categories using high threshold T_H and low threshold T_L (T_L = T_H / 3):
        - Strong edge: |E_YX> = |10> (G >= T_H)
        - Weak edge:   |E_YX> = |01> (T_L <= G < T_H)
        - Non-edge:    |E_YX> = |00> (G < T_L)

    Equation (14):
        |E> = (1 / 2^n) sum_{Y=0}^{2^n-1} sum_{X=0}^{2^n-1} |E_YX> |Y> |X>
        where |E_YX> = |E_1 E_0> in {00, 01, 10}.

    Inputs:
        q_bits (int): Bit length of threshold registers.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing double threshold classification.

    Time Complexity:
        O(q_bits) gates.

    Space Complexity:
        3 * q_bits (G, T_H, T_L) + 2 output edge qubits |E_1 E_0> + comparator aux qubits.

    Reference:
        Paper Section 3.2, Step 5, Equation (14), Figure 18.
    """
    g_reg = QuantumRegister(q_bits, name='g_val')
    th_high = QuantumRegister(q_bits, name='t_high')
    th_low = QuantumRegister(q_bits, name='t_low')
    e_out = QuantumRegister(2, name='edge_e1_e0')  # e_out[1] = E_1, e_out[0] = E_0
    comp_flags = QuantumRegister(4, name='comp_flags')
    aux_comp = QuantumRegister(q_bits, name='aux_comp')

    qc = QuantumCircuit(g_reg, th_high, th_low, e_out, comp_flags, aux_comp, name='Double_Threshold_Circuit')

    qc_comp = build_quantum_comparator(q_bits)

    # QC 1: Compare G and T_H -> comp_flags[1] is (G > T_H), comp_flags[0] is (G < T_H)
    qc.append(qc_comp, list(g_reg) + list(th_high) + [comp_flags[1], comp_flags[0]] + list(aux_comp))

    # QC 2: Compare G and T_L -> comp_flags[3] is (G > T_L), comp_flags[2] is (G < T_L)
    qc.append(qc_comp, list(g_reg) + list(th_low) + [comp_flags[3], comp_flags[2]] + list(aux_comp))

    # 1. Strong edge (E_1 = 1, E_0 = 0): Activated if G >= T_H -> NOT(G < T_H) i.e. comp_flags[0] == 0
    qc.x(comp_flags[0])
    qc.cx(comp_flags[0], e_out[1])
    qc.x(comp_flags[0])  # Restore

    # 2. Weak edge (E_1 = 0, E_0 = 1): Activated if T_L <= G < T_H
    # i.e. NOT(G < T_L) AND (G < T_H) -> comp_flags[2] == 0 AND comp_flags[0] == 1
    qc.x(comp_flags[2])
    qc.ccx(comp_flags[2], comp_flags[0], e_out[0])
    qc.x(comp_flags[2])  # Restore

    return qc
