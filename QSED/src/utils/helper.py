"""
Helper Utilities Module for Quantum Image Edge Detection (QSED).

Provides string-integer bit conversions, bit padding, and binary arithmetic utilities.
"""

def int_to_bitstring(val: int, num_bits: int) -> str:
    """
    Convert an integer to a zero-padded binary string representation of length num_bits.
    """
    return format(val, f'0{num_bits}b')


def bitstring_to_int(bitstr: str) -> int:
    """
    Convert a binary string representation to an integer.
    """
    return int(bitstr, 2)
