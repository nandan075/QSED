"""
Quantum sub-modules for Quantum Image Edge Detection (QSED).

Contains basic quantum logic blocks (Comparator, Cycle Shift, Parallel Adder,
Complement, Absolute Value, Double Operation, Copy Operation) and complete pipeline circuits.
"""

from .neqr import NEQRImage, prepare_neqr_with_auxiliary
from .comparator import build_quantum_comparator
from .cycle_shift import build_cycle_shift
from .parallel_adder import build_parallel_adder
from .complement import build_complement_operation
from .absolute_value import build_absolute_value
from .double_operation import build_double_operation
from .copy_operation import build_copy_operation

__all__ = [
    "NEQRImage",
    "prepare_neqr_with_auxiliary",
    "build_quantum_comparator",
    "build_cycle_shift",
    "build_parallel_adder",
    "build_complement_operation",
    "build_absolute_value",
    "build_double_operation",
    "build_copy_operation",
]
