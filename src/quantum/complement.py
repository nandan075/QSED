"""
Complement Operation (CA) Module for Quantum Image Edge Detection (QSED).

Calculates two's complement representation for signed binary integers.

Paper Reference:
- Section 3.1 (4): Quantum absolute value operation (Equation 6, Figure 7, Ref. [34-37]).
"""

from qiskit import QuantumCircuit, QuantumRegister
from .cycle_shift import build_cycle_shift


def build_complement_operation(n_bits: int) -> QuantumCircuit:
    """
    Construct Complement Operation (CA) quantum circuit.

    Purpose:
        Compute two's complement [x]_{CA} for signed binary integers (Eq. 6).
        Used to perform subtraction via addition: A - B = A + (~B + 1) per Equation (7).

    Equation (6):
        [x]_{CA} = 0 x_{n-1} ... x_0, if x_n = 0
        [x]_{CA} = 1 \bar{x}_{n-1} ... \bar{x}_0 + 1, if x_n = 1

    Inputs:
        n_bits (int): Bit length of value register (excluding sign bit).

    Outputs:
        QuantumCircuit: Qiskit quantum circuit for CA block.

    Time Complexity:
        O(n_bits) gates.

    Space Complexity:
        n_bits value qubits + 1 sign qubit.

    Reference:
        Paper Section 3.1 (4), Equation (6), Figure 7.
    """
    val_reg = QuantumRegister(n_bits, name='val')
    sign_bit = QuantumRegister(1, name='sign')

    qc = QuantumCircuit(val_reg, sign_bit, name=f'CA_{n_bits}bit')

    # If sign_bit == 1 (negative number):
    # 1. Bitwise NOT on value bits: apply X gates controlled by sign_bit
    for i in range(n_bits):
        qc.cx(sign_bit[0], val_reg[i])

    # 2. Add 1 to value bits: controlled CT(+1) shift on val_reg if sign_bit == 1
    ct_inc = build_cycle_shift(n_bits, direction=+1)
    ctrl_ct = ct_inc.to_gate().control(1)
    qc.append(ctrl_ct, [sign_bit[0]] + list(val_reg))

    return qc
