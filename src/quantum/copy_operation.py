"""
Quantum Copy Operation Module for Quantum Image Edge Detection (QSED).

Duplicates the state of a source qubit register onto a target auxiliary register using CNOT gates.

Paper Reference:
- Section 3.1 (6): Quantum copy operation (Figure 10, Ref. [34] Iliyasu et al.).
- Section 3.2: Step 2 Quantum image set shift transformation.
"""

from qiskit import QuantumCircuit, QuantumRegister


def build_copy_operation(n_bits: int) -> QuantumCircuit:
    """
    Construct Quantum Copy Operation circuit.

    Purpose:
        Copy grayscale values |C_YX> of shifted image pixels into prepared auxiliary qubits
        during Step 2 (Shift Transformation) to build the 24 neighbor images in quantum superposition.

    Inputs:
        n_bits (int): Bit length of source and target registers.

    Outputs:
        QuantumCircuit: Qiskit circuit with CNOT gates connecting source and target registers.

    Time Complexity:
        O(n_bits) CNOT gate operations.

    Space Complexity:
        2 * n_bits qubits (source + target).

    Reference:
        Paper Section 3.1 (6), Figure 10.
    """
    source = QuantumRegister(n_bits, name='source')
    target = QuantumRegister(n_bits, name='target_aux')

    qc = QuantumCircuit(source, target, name=f'Copy_{n_bits}bit')

    # Apply CNOT from each source qubit to target auxiliary qubit
    for i in range(n_bits):
        qc.cx(source[i], target[i])

    return qc
