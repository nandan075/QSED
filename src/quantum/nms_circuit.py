"""
Non-Maximum Suppression (NMS) Quantum Circuit Module for QSED.

Designs the quantum circuit for Non-Maximum Suppression per Equation (13) and Figure 17.

Paper Reference:
- Section 3.2: Step 4 Non-maximum suppression (Equation 13, Figure 17).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .comparator import build_quantum_comparator


def build_nms_circuit(q_bits: int = 8) -> QuantumCircuit:
    """
    Construct Non-Maximum Suppression quantum circuit.

    Purpose:
        Compare current pixel gradient |G(Y,X)> with two neighboring pixel gradients |G_n1> and |G_n2>
        along the dominant gradient orientation using two Quantum Comparators (QC).
        If |G(Y,X)> >= |G_n1> and |G(Y,X)> >= |G_n2>, set mask qubit |M> = |1> (maximum retained pixel).
        Otherwise, set |M> = |0> (suppressed pixel).

    Equation (13):
        |G_S> = (1 / 2^n) sum_{Y=0}^{2^n-1} sum_{X=0}^{2^n-1} |M> |G> |Y> |X>

    Inputs:
        q_bits (int): Bit width of gradient value.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing NMS filtering logic.

    Time Complexity:
        O(q_bits) gate operations.

    Space Complexity:
        3 * q_bits input + 1 output mask qubit |M> + comparator aux qubits.

    Reference:
        Paper Section 3.2, Step 4, Equation (13), Figure 17.
    """
    g_curr = QuantumRegister(q_bits, name='g_curr')
    g_neigh1 = QuantumRegister(q_bits, name='g_neigh1')
    g_neigh2 = QuantumRegister(q_bits, name='g_neigh2')
    m_flag = QuantumRegister(1, name='m_max_flag')  # |M> = 1 if max, 0 if suppressed
    c_flags = QuantumRegister(4, name='c_comp_flags')  # Flags from 2 QC comparators
    aux_comp = QuantumRegister(q_bits, name='aux_comp')

    qc = QuantumCircuit(g_curr, g_neigh1, g_neigh2, m_flag, c_flags, aux_comp, name='NMS_Circuit')

    qc_comp = build_quantum_comparator(q_bits)

    # QC 1: Compare g_curr and g_neigh1 -> Output c_flags[1] (A>B), c_flags[0] (A<B)
    qc.append(qc_comp, list(g_curr) + list(g_neigh1) + [c_flags[1], c_flags[0]] + list(aux_comp))

    # QC 2: Compare g_curr and g_neigh2 -> Output c_flags[3] (A>B), c_flags[2] (A<B)
    qc.append(qc_comp, list(g_curr) + list(g_neigh2) + [c_flags[3], c_flags[2]] + list(aux_comp))

    # g_curr is local maximum if NOT(g_curr < g_neigh1) AND NOT(g_curr < g_neigh2)
    # i.e. c_flags[0] == 0 AND c_flags[2] == 0
    qc.x(c_flags[0])
    qc.x(c_flags[2])
    qc.ccx(c_flags[0], c_flags[2], m_flag[0])
    qc.x(c_flags[0])  # Restore flag
    qc.x(c_flags[2])

    return qc
