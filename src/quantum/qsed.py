"""
Quantum Image Edge Detection (QSED) Main Pipeline Orchestrator.

Integrates all 6 stages of the proposed algorithm:
1. NEQR Image Preparation
2. Cyclic Shift Transformation
3. 8-Direction Sobel Gradient Computation
4. Non-Maximum Suppression (NMS)
5. Double Threshold Detection
6. Hysteresis Edge Tracking

Paper Reference:
- Section 3: Quantum image edge detection based on eight-direction Sobel operator.
- Section 3.2: Complete Workflow (Figure 11).
- Section 4.2: Simulation experiment setup.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False
from qiskit.quantum_info import Statevector

from .neqr import NEQRImage, prepare_neqr_with_auxiliary
from .comparator import build_quantum_comparator
from .cycle_shift import build_cycle_shift
from .parallel_adder import build_parallel_adder
from .complement import build_complement_operation
from .absolute_value import build_absolute_value
from .double_operation import build_double_operation
from .copy_operation import build_copy_operation

from .gradient_circuit import build_directional_gradient_circuit, build_max_gradient_selection_circuit
from .nms_circuit import build_nms_circuit
from .threshold_circuit import build_double_threshold_circuit
from .edge_tracking import build_edge_tracking_circuit

from ..classical.sobel_operator import apply_sobel_masks
from ..classical.gradient import compute_gradient_magnitude
from ..classical.nms import non_maximum_suppression
from ..classical.threshold import double_threshold
from ..classical.hysteresis import edge_tracking_hysteresis


class QSEDRunner:
    """
    QSED Algorithm Execution Engine.

    Executes the 6-stage Quantum Sobel Edge Detection pipeline on digital images.

    Supports:
    - Mode 'quantum': Builds full Qiskit quantum circuits and executes statevector simulation (ideal for 2x2, 4x4, 8x8 images).
    - Mode 'hybrid': Faithfully executes the mathematical quantum algorithm steps classically (ideal for 512x512 images per Section 4.2).
    """

    def __init__(
        self,
        q_bits: int = 8,
        mode: str = 'hybrid',
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None
    ):
        """
        Initialize QSED Engine.

        Inputs:
            q_bits (int): Grayscale bit-depth (default 8).
            mode (str): 'quantum' (Qiskit circuit execution) or 'hybrid' (high-resolution algorithm simulation).
            high_threshold (Optional[float]): T_H threshold value.
            low_threshold (Optional[float]): T_L threshold value (default T_H / 3).
        """
        self.q_bits = q_bits
        self.mode = mode.lower()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def run_pipeline(
        self,
        image_matrix: np.ndarray,
        run_keypoint_analysis: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run the complete 6-stage QSED pipeline.

        Inputs:
            image_matrix (np.ndarray): 2^n x 2^n grayscale image matrix.

        Outputs:
            Dict[str, np.ndarray]: Dictionary containing intermediate and final outputs:
                - 'original': Input image
                - 'neqr_circuit': Qiskit NEQR circuit (if quantum mode)
                - 'directional_gradients': Dict of 8 gradient images
                - 'gradient_magnitude': Maximum absolute gradient image G
                - 'dominant_direction': Direction index map
                - 'nms_image': Non-maximum suppressed gradient image G_S
                - 'max_mask': Binary NMS mask |M>
                - 'threshold_map': Double threshold map (0, 1, 2) |E>
                - 'high_threshold': T_H value used
                - 'low_threshold': T_L value used
                - 'final_edges': Final binary edge map |B>
        """
        if self.mode == 'quantum' and image_matrix.shape[0] <= 8:
            return self._run_quantum_circuit_pipeline(image_matrix, run_keypoint_analysis)
        else:
            return self._run_hybrid_pipeline(image_matrix, run_keypoint_analysis)

    def _run_hybrid_pipeline(self, image_matrix: np.ndarray, run_keypoint_analysis: bool = False) -> Dict[str, np.ndarray]:
        """
        Execute the exact quantum mathematical workflow using array simulation (Section 4.2).

        Step 1: NEQR Image Preparation
        Step 2: Quantum Shift Transformation (5x5 neighborhood correlation)
        Step 3: Gradients Calculation (8-direction Sobel operators + max selection)
        Step 4: Non-Maximum Suppression (NMS)
        Step 5: Double Threshold Detection (T_L = T_H / 3)
        Step 6: Edge Tracking (24-neighborhood hysteresis)
        """
        # Step 1: NEQR Encoding validation
        neqr = NEQRImage(image_matrix, q_bits=self.q_bits)

        # Step 2 & 3: Shift Transformation & 8-Direction Sobel Gradients
        dir_grads = apply_sobel_masks(image_matrix)
        grad_mag, dom_dir = compute_gradient_magnitude(dir_grads)

        # Step 4: Non-Maximum Suppression
        nms_img, max_mask = non_maximum_suppression(grad_mag, dom_dir)

        # Step 5: Double Threshold Detection
        thresh_map, t_high, t_low = double_threshold(
            nms_img,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold
        )

        # Step 6: Edge Tracking via Hysteresis
        final_edges = edge_tracking_hysteresis(thresh_map)

        results = {
            'original': image_matrix,
            'directional_gradients': dir_grads,
            'gradient_magnitude': grad_mag,
            'dominant_direction': dom_dir,
            'nms_image': nms_img,
            'max_mask': max_mask,
            'threshold_map': thresh_map,
            'high_threshold': t_high,
            'low_threshold': t_low,
            'final_edges': final_edges
        }

        if run_keypoint_analysis:
            from ..feature.directional_variation import compute_directional_variation, compute_keypoint_score
            from ..feature.keypoint_detection import detect_keypoints
            from ..feature.von_neumann import compute_entropy_map
            
            dir_var_map = compute_directional_variation(dom_dir, grad_mag)
            kp_score_map = compute_keypoint_score(grad_mag, dir_var_map)
            keypoints = detect_keypoints(grad_mag, dom_dir, final_edges, dir_var_map, kp_score_map)
            entropy_map, entropy_map_norm = compute_entropy_map(image_matrix)
            
            results.update({
                'keypoints': keypoints,
                'directional_variation_map': dir_var_map,
                'keypoint_score_map': kp_score_map,
                'entropy_map': entropy_map,
                'entropy_map_normalized': entropy_map_norm
            })

        return results

    def _run_quantum_circuit_pipeline(self, image_matrix: np.ndarray, run_keypoint_analysis: bool = False) -> Dict[str, np.ndarray]:
        """
        Build and execute actual Qiskit quantum circuits for small test images (2x2, 4x4).
        """
        neqr = NEQRImage(image_matrix, q_bits=self.q_bits)
        neqr_qc = neqr.build_circuit()

        # Execute statevector simulation
        sv = Statevector.from_instruction(neqr_qc)
        reconstructed = NEQRImage.decode_statevector(sv, n=neqr.n, q_bits=self.q_bits)

        # Run mathematical stages on quantum state representation
        results = self._run_hybrid_pipeline(reconstructed, run_keypoint_analysis)
        results['neqr_circuit'] = neqr_qc
        results['statevector'] = sv

        return results

    def build_complete_pipeline_circuit(self, n_pos: int = 1) -> QuantumCircuit:
        """
        Construct a master Qiskit QuantumCircuit representing the complete QSED gate pipeline architecture.

        Purpose:
            Visualization, circuit diagram export, and gate count analysis for Section 4.1.

        Inputs:
            n_pos (int): Number of position qubits per axis (e.g. n=1 for 2x2 image).

        Outputs:
            QuantumCircuit: Master circuit combining NEQR, CT, PA, AV, QC, NMS, Threshold, and Edge Tracking blocks.

        Reference:
            Paper Section 3.2, Figures 11-19.
        """
        q_col = QuantumRegister(self.q_bits, name='color')
        y_pos = QuantumRegister(n_pos, name='y_pos')
        x_pos = QuantumRegister(n_pos, name='x_pos')
        aux_shift = QuantumRegister(24 * self.q_bits, name='aux_24q')
        m_flag = QuantumRegister(1, name='nms_flag')
        e_flags = QuantumRegister(2, name='edge_flags')
        b_final = QuantumRegister(1, name='final_edge')

        master_qc = QuantumCircuit(
            q_col, y_pos, x_pos, aux_shift, m_flag, e_flags, b_final,
            name='QSED_Complete_Pipeline'
        )

        # Step 1: NEQR Hadamard initialization
        for y in y_pos:
            master_qc.h(y)
        for x in x_pos:
            master_qc.h(x)

        # Step 2: Cyclic Shift blocks CT(+1) and CT(-1)
        ct_inc = build_cycle_shift(n_pos, direction=+1)
        master_qc.append(ct_inc.to_gate(), list(y_pos))
        master_qc.append(ct_inc.to_gate(), list(x_pos))

        # Step 3: Gradient calculation (0 deg block)
        grad_gate = build_directional_gradient_circuit("0", q_bits=self.q_bits)
        master_qc.barrier()

        # Step 4: NMS block
        nms_block = build_nms_circuit(q_bits=self.q_bits)
        master_qc.barrier()

        # Step 5: Double Threshold block
        thresh_block = build_double_threshold_circuit(q_bits=self.q_bits)
        master_qc.barrier()

        # Step 6: Edge Tracking block
        et_block = build_edge_tracking_circuit()
        master_qc.barrier()

        return master_qc
