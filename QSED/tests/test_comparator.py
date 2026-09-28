"""
Unit Tests for Quantum Comparator (QC) Module.

Tests bitwise comparisons A > B, A < B, and A == B.
Paper Reference: Section 3.1 (1), Figure 4.
"""

import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from src.quantum.comparator import build_quantum_comparator


def test_quantum_comparator_logic():
    """
    Verify QC circuit outputs correct flags C_1 (A>B) and C_0 (A<B).
    """
    n_bits = 2
    qc_comp = build_quantum_comparator(n_bits)

    # Test cases: (A, B, Expected C1, Expected C0)
    test_cases = [
        (3, 1, 1, 0),  # 3 > 1 -> C1=1, C0=0
        (1, 3, 0, 1),  # 1 < 3 -> C1=0, C0=1
        (2, 2, 0, 0),  # 2 == 2 -> C1=0, C0=0
    ]

    for a_val, b_val, exp_c1, exp_c0 in test_cases:
        a_reg = QuantumRegister(n_bits, name='a')
        b_reg = QuantumRegister(n_bits, name='b')
        c_out = QuantumRegister(2, name='c_out')
        aux = QuantumRegister(n_bits, name='aux')

        test_qc = QuantumCircuit(a_reg, b_reg, c_out, aux)

        # Prepare A and B states
        a_bin = format(a_val, f'0{n_bits}b')
        b_bin = format(b_val, f'0{n_bits}b')

        for i, b in enumerate(reversed(a_bin)):
            if b == '1':
                test_qc.x(a_reg[i])

        for i, b in enumerate(reversed(b_bin)):
            if b == '1':
                test_qc.x(b_reg[i])

        # Append QC
        test_qc.append(qc_comp, list(a_reg) + list(b_reg) + list(c_out) + list(aux))

        # Evaluate statevector
        sv = Statevector.from_instruction(test_qc)
        state_dict = sv.to_dict()

        # Find active non-zero state
        active_bitstring = None
        for bstr, amp in state_dict.items():
            if abs(amp) > 1e-5:
                active_bitstring = bstr
                break

        # Register layout in bitstring (Qiskit reverses order):
        # aux (n), c_out (2), b (n), a (n)
        # c_out[1] is C1, c_out[0] is C0
        # In bitstring, c_out starts at index n (from right): aux(n) c_out(2) b(n) a(n)
        # So c_out bits are at positions: len - n - 2 to len - n
        total_qubits = 2 * n_bits + 2 + n_bits
        clean_str = active_bitstring.zfill(total_qubits)

        # c_out register is c_out[1] c_out[0]
        # Qiskit registers: aux (n_bits), c_out (2 bits), b_reg (n_bits), a_reg (n_bits)
        # In Qiskit string representation: aux_reg is leftmost, a_reg is rightmost
        c_out_substr = clean_str[n_bits:n_bits+2]
        meas_c1 = int(c_out_substr[0])
        meas_c0 = int(c_out_substr[1])

        assert (meas_c1, meas_c0) == (exp_c1, exp_c0), \
            f"Failed for A={a_val}, B={b_val}. Got C1={meas_c1}, C0={meas_c0}, expected {exp_c1}, {exp_c0}"
