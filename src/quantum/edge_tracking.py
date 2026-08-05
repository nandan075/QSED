"""
Edge Tracking Quantum Circuit Module for QSED.

Designs the quantum circuit for hysteresis edge tracking per Equation (15) and Figure 19.

Paper Reference:
- Section 3.2: Step 6 Edge tracking (Equation 15, Figure 19).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .comparator import build_quantum_comparator


def build_edge_tracking_circuit() -> QuantumCircuit:
    """
    Construct Edge Tracking quantum circuit.

    Purpose:
        Evaluate weak candidate edges (|E_YX> = |01>). If any pixel in its 24-neighborhood (5x5 window)
        is a strong edge (|E_neigh> = |10>), set final edge qubit |B_YX> = |1>.
        Otherwise, set |B_YX> = |0>.

    Equation (15):
        |B> = (1 / 2^n) sum_{Y=0}^{2^n-1} sum_{X=0}^{2^n-1} |B_YX> |Y> |X>
        where |B_YX> in {0, 1}.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing hysteresis edge tracking.

    Time Complexity:
        O(24 * q_bits) gate operations.

    Space Complexity:
        24 * 2 neighbor threshold qubits + 2 current pixel threshold qubits + 1 output qubit |B_YX>.

    Reference:
        Paper Section 3.2, Step 6, Equation (15), Figure 19.
    """
    e_curr = QuantumRegister(2, name='e_curr_pixel')  # Current pixel threshold state |E_YX>
    n_strong = QuantumRegister(24, name='neighbor_has_strong')  # 24 flag qubits for neighborhood strong edges
    b_out = QuantumRegister(1, name='final_edge_b')  # |B_YX> output

    qc = QuantumCircuit(e_curr, n_strong, b_out, name='Edge_Tracking_Circuit')

    # 1. If current pixel is already strong edge (e_curr == |10>, i.e. e_curr[1]==1, e_curr[0]==0),
    # set b_out = 1 directly
    qc.x(e_curr[0])
    qc.ccx(e_curr[1], e_curr[0], b_out[0])
    qc.x(e_curr[0])  # Restore

    # 2. If current pixel is weak edge (e_curr == |01>, i.e. e_curr[1]==0, e_curr[0]==1) AND
    # at least one n_strong[k] == 1, set b_out = 1
    # Multi-control or OR reduction over 24 neighborhood flags
    qc.x(e_curr[1])
    for k in range(24):
        qc.mcx([e_curr[1], e_curr[0], n_strong[k]], b_out[0])
    qc.x(e_curr[1])  # Restore

    return qc
