"""
Pytest configuration for QCSI: adds the QCSI root to sys.path.
"""

import sys
import os

qcsi_root = os.path.dirname(os.path.abspath(__file__))
if qcsi_root not in sys.path:
    sys.path.insert(0, qcsi_root)
