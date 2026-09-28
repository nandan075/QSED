"""
Circuit Diagram Generator Script for Quantum Image Edge Detection (QSED).

Generates and exports quantum circuit diagrams for:
- NEQR Image Encoding (Fig. 1)
- Quantum Comparator QC (Fig. 4)
- Cycle Shift CT(+1) and CT(-1) (Fig. 5)
- Reversible Parallel Adder PA (Fig. 6)
- Complement Operation CA (Fig. 7)
- Quantum Absolute Value AV (Fig. 8)
- Quantum Double Operation DO (Fig. 9)
- Quantum Copy Operation (Fig. 10)
- Gradient Calculation Circuits (Figs. 12-16)
- Non-Maximum Suppression Circuit (Fig. 17)
- Double Threshold Detection Circuit (Fig. 18)
- Edge Tracking Circuit (Fig. 19)
- Complete QSED Master Pipeline Circuit (Fig. 11)

Outputs stored in docs/circuits/ as PNG and Text representations.

Paper Reference: Section 3.1 & 3.2.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qiskit import QuantumCircuit
from src.quantum.neqr import NEQRImage
from src.quantum.comparator import build_quantum_comparator
from src.quantum.cycle_shift import build_cycle_shift
from src.quantum.parallel_adder import build_parallel_adder
from src.quantum.complement import build_complement_operation
from src.quantum.absolute_value import build_absolute_value
from src.quantum.double_operation import build_double_operation
from src.quantum.copy_operation import build_copy_operation
from src.quantum.gradient_circuit import build_directional_gradient_circuit, build_max_gradient_selection_circuit
from src.quantum.nms_circuit import build_nms_circuit
from src.quantum.threshold_circuit import build_double_threshold_circuit
from src.quantum.edge_tracking import build_edge_tracking_circuit
from src.quantum.qsed import QSEDRunner


def export_circuit(qc: QuantumCircuit, filename: str, docs_dir: str):
    """
    Save circuit diagrams as text art (.txt) and matplotlib rendering (.png).
    """
    txt_path = os.path.join(docs_dir, f"{filename}.txt")
    png_path = os.path.join(docs_dir, f"{filename}.png")

    # Export text diagram
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(qc.draw(output='text')))

    # Export PNG diagram using matplotlib
    try:
        fig = qc.draw(output='mpl', style='iqp')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Notice: Matplotlib plot export for {filename} used fallback text layout. ({e})")


def main():
    docs_circuits = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "circuits"))
    os.makedirs(docs_circuits, exist_ok=True)
    print(f"Generating quantum circuits in: {docs_circuits}")

    # 1. NEQR 2x2 Circuit
    sample_img = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    neqr = NEQRImage(sample_img, q_bits=8)
    export_circuit(neqr.build_circuit(), "neqr_2x2_circuit", docs_circuits)

    # 2. Quantum Comparator QC (2-bit)
    qc_comp = build_quantum_comparator(n_bits=2)
    export_circuit(qc_comp, "quantum_comparator_qc", docs_circuits)

    # 3. Cycle Shift CT(+1) and CT(-1) (2-bit)
    export_circuit(build_cycle_shift(2, direction=+1), "cycle_shift_plus1", docs_circuits)
    export_circuit(build_cycle_shift(2, direction=-1), "cycle_shift_minus1", docs_circuits)

    # 4. Reversible Parallel Adder PA (2-bit)
    export_circuit(build_parallel_adder(2), "parallel_adder_pa", docs_circuits)

    # 5. Complement Operation CA (2-bit)
    export_circuit(build_complement_operation(2), "complement_operation_ca", docs_circuits)

    # 6. Quantum Absolute Value AV (2-bit)
    export_circuit(build_absolute_value(2), "absolute_value_av", docs_circuits)

    # 7. Quantum Double Operation DO (2-bit)
    export_circuit(build_double_operation(2), "double_operation_do", docs_circuits)

    # 8. Quantum Copy Operation (2-bit)
    export_circuit(build_copy_operation(2), "copy_operation", docs_circuits)

    # 9. Directional Gradient Circuit (0° direction)
    export_circuit(build_directional_gradient_circuit("0", q_bits=4), "gradient_0deg_circuit", docs_circuits)

    # 10. Max Gradient Selection Tree
    export_circuit(build_max_gradient_selection_circuit(q_bits=4), "max_gradient_selection", docs_circuits)

    # 11. Non-Maximum Suppression (NMS) Circuit
    export_circuit(build_nms_circuit(q_bits=4), "nms_circuit", docs_circuits)

    # 12. Double Threshold Detection Circuit
    export_circuit(build_double_threshold_circuit(q_bits=4), "double_threshold_circuit", docs_circuits)

    # 13. Edge Tracking Circuit
    export_circuit(build_edge_tracking_circuit(), "edge_tracking_circuit", docs_circuits)

    # 14. Complete Master Pipeline Architecture Circuit
    runner = QSEDRunner(q_bits=4, mode='quantum')
    master_qc = runner.build_complete_pipeline_circuit(n_pos=1)
    export_circuit(master_qc, "qsed_complete_pipeline", docs_circuits)

    print("All circuit diagrams successfully generated and exported to docs/circuits/")


if __name__ == "__main__":
    main()
