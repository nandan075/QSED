"""
Unit Tests for Cycle Shift Transformation (CT) Quantum Operations.

Tests CT(+1) increment and CT(-1) decrement modulo 2^n.
Paper Reference: Section 3.1 (2), Figure 5.
"""

import pytest
from qiskit.quantum_info import Statevector
from src.quantum.cycle_shift import build_cycle_shift


def test_cycle_shift_plus_one():
    """
    Test CT(+1) on 2-qubit register: |0> -> |1>, |1> -> |2>, |2> -> |3>, |3> -> |0>.
    """
    ct = build_cycle_shift(n_bits=2, direction=+1)

    # Initial state |00>
    sv0 = Statevector.from_int(0, dims=4)
    sv1 = sv0.evolve(ct)
    # Binary bit order: state 1 is |01>
    assert list(sv1.to_dict().keys()) == ['01']

    # Initial state |11> (3) -> should wrap to |00> (0)
    sv3 = Statevector.from_int(3, dims=4)
    sv_wrapped = sv3.evolve(ct)
    assert list(sv_wrapped.to_dict().keys()) == ['00']


def test_cycle_shift_minus_one():
    """
    Test CT(-1) on 2-qubit register: |0> -> |3>, |1> -> |0>, |2> -> |1>, |3> -> |2>.
    """
    ct_minus = build_cycle_shift(n_bits=2, direction=-1)

    # Initial state |00> (0) -> wrap to |11> (3)
    sv0 = Statevector.from_int(0, dims=4)
    sv_wrapped = sv0.evolve(ct_minus)
    assert list(sv_wrapped.to_dict().keys()) == ['11']

    # Initial state |01> (1) -> |00> (0)
    sv1 = Statevector.from_int(1, dims=4)
    sv0_res = sv1.evolve(ct_minus)
    assert list(sv0_res.to_dict().keys()) == ['00']
