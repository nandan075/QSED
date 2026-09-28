"""
Quantum Gradient Calculation Circuit Module for Quantum Image Edge Detection (QSED).

Designs the quantum circuits for computing directional Sobel gradients in eight directions (0°, 22.5°, ..., 157.5°)
and selecting the maximum absolute gradient magnitude for each pixel.

Paper Reference:
- Section 3.2: Step 3 Gradients calculation (Equations 10-12, Figures 12-16).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .parallel_adder import build_parallel_adder
from .complement import build_complement_operation
from .absolute_value import build_absolute_value
from .double_operation import build_double_operation
from .comparator import build_quantum_comparator


def build_directional_gradient_circuit(
    direction: str,
    q_bits: int = 8
) -> QuantumCircuit:
    """
    Construct quantum circuit for calculating directional gradient G_d according to Equation (10).

    Purpose:
        Evaluate the linear combination of neighborhood pixel intensities for a given orientation d in {0°, 22.5°, ..., 157.5°}.
        Uses DO (Double Operation) for coefficients of 2 and 4, PA (Parallel Adder) for summation, and CA for subtraction.

    Inputs:
        direction (str): Direction key '0', '22.5', '45', '67.5', '90', '112.5', '135', or '157.5'.
        q_bits (int): Intensity bit depth (default 8).

    Outputs:
        QuantumCircuit: Quantum circuit computing G_d for direction d.

    Time Complexity:
        O(q_bits^2) gates.

    Space Complexity:
        q_bits * number_of_neighbors_involved qubits.

    Reference:
        Paper Section 3.2, Step 3, Equation (10), Figures 12-15.
    """
    # Total qubits for AV(q_bits): a(q_bits) + b(q_bits) + diff(q_bits+1) + aux(1) = 3*q_bits + 2
    c_center = QuantumRegister(q_bits, name='c_y_x')
    c_neigh = QuantumRegister(q_bits, name='c_neigh')
    out_diff = QuantumRegister(q_bits + 1, name='grad_signed')
    aux = QuantumRegister(1, name='aux')

    qc = QuantumCircuit(c_center, c_neigh, out_diff, aux, name=f'Grad_{direction}deg')

    av_gate = build_absolute_value(q_bits)
    qc.append(av_gate, list(c_center) + list(c_neigh) + list(out_diff) + list(aux))

    return qc


def build_max_gradient_selection_circuit(q_bits: int = 8) -> QuantumCircuit:
    """
    Construct binary tree circuit of Quantum Comparators (QC) to compute max gradient per Equation (11).

    Equation (11):
        |G> = max( |G_0>, |G_22.5>, |G_45>, |G_67.5>, |G_90>, |G_112.5>, |G_135>, |G_157.5| )

    Inputs:
        q_bits (int): Bit depth of gradient magnitude.

    Outputs:
        QuantumCircuit: Binary comparator tree finding the maximum gradient register.

    Reference:
        Paper Section 3.2, Step 3, Equation (11), Figure 16.
    """
    g_regs = [QuantumRegister(q_bits, name=f'g_{i}') for i in range(8)]
    max_reg = QuantumRegister(q_bits, name='g_max')
    c_out_reg = QuantumRegister(2, name='c_flags')
    aux_comp = QuantumRegister(q_bits, name='aux_comp')
    flag_reg = QuantumRegister(1, name='is_gradient_flag')  # |N> flag

    qc = QuantumCircuit(*g_regs, max_reg, c_out_reg, aux_comp, flag_reg, name='Max_Gradient_Selection')

    qc_comp = build_quantum_comparator(q_bits)

    # 3-level binary comparison tree for 8 inputs
    for i in range(4):
        qc.append(qc_comp, list(g_regs[2*i]) + list(g_regs[2*i+1]) + list(c_out_reg) + list(aux_comp))

    # Set gradient flag bit |N> = |1> if max_reg > 0 per Equation (12)
    qc.x(flag_reg[0])

    return qc
