"""
Cycle Shift Transformation (CT) Module for Quantum Image Edge Detection (QSED).

Implements spatial shift operations CT(+1) and CT(-1) on n-qubit position registers.

Paper Reference:
- Section 3.1 (2): Cycle shift transformation operation (Figure 5, Ref. [15, 31, 32]).
- Section 3.2: Step 2 Quantum image set shift transformation (Table 1).
"""

from qiskit import QuantumCircuit, QuantumRegister


def build_cycle_shift(n_bits: int, direction: int = +1) -> QuantumCircuit:
    """
    Construct Cyclic Shift Transformation (CT) quantum circuit for an n-qubit position register.

    Purpose:
        Perform modular spatial translations |(Y + 1) mod 2^n> or |(Y - 1) mod 2^n>.
        This shifts the entire image simultaneously by one pixel unit along X or Y axis in quantum superposition,
        enabling parallel 5x5 neighborhood pixel alignment.

    Inputs:
        n_bits (int): Bit length of position register |Y> or |X>.
        direction (int): +1 for CT(+1) shift, -1 for CT(-1) shift.

    Outputs:
        QuantumCircuit: Qiskit quantum circuit implementing CT operation.

    Time Complexity:
        O(n_bits) gate operations.

    Space Complexity:
        n_bits qubits (in-place operation).

    Reference:
        Paper Section 3.1 (2), Figure 5.
    """
    pos_reg = QuantumRegister(n_bits, name='pos')
    qc = QuantumCircuit(pos_reg, name=f"CT({'%+d' % direction})")

    if direction == +1:
        # CT(+1): Quantum incrementer circuit
        # Multi-controlled NOT gates from LSB pos[0] to MSB pos[n-1]
        for i in reversed(range(n_bits)):
            controls = list(pos_reg[:i])
            target = pos_reg[i]
            if len(controls) == 0:
                qc.x(target)
            elif len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.mcx(controls, target)

    elif direction == -1:
        # CT(-1): Quantum decrementer circuit using the identity CT(-1) = X^n * CT(+1) * X^n
        for i in range(n_bits):
            qc.x(pos_reg[i])

        # Apply CT(+1) gate
        ct_inc = build_cycle_shift(n_bits, direction=+1)
        qc.append(ct_inc.to_gate(), list(pos_reg))

        for i in range(n_bits):
            qc.x(pos_reg[i])

    else:
        raise ValueError(f"Direction must be +1 or -1, got {direction}")

    return qc
