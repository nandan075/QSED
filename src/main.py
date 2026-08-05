"""
Main Entry Point for Quantum Image Edge Detection (QSED).

Command line interface for executing the 8-direction QSED pipeline on input images.

Usage:
    python -m src.main --image images/lena.png --mode hybrid --output outputs/final/
    python -m src.main --demo --mode quantum
    python -m src.main --benchmark

Paper Reference: Section 3 & 4.
"""

import argparse
import os
import sys
import numpy as np

from src.classical.image_loader import load_image, save_image, generate_synthetic_image
from src.quantum.qsed import QSEDRunner
from src.utils.visualization import plot_qsed_pipeline_results, compare_sobel_methods
from src.utils.metrics import calculate_mse


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Image Edge Detection (QSED) based on Eight-Direction Sobel Operator"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image file.")
    parser.add_argument("--size", type=int, default=512, help="Image size (must be power of 2, e.g. 16, 64, 512).")
    parser.add_argument("--mode", type=str, choices=['quantum', 'hybrid'], default='hybrid',
                        help="Execution mode: 'quantum' (Qiskit circuit sim for small images) or 'hybrid' (high-res simulation).")
    parser.add_argument("--output", type=str, default="outputs/final/", help="Output directory path.")
    parser.add_argument("--demo", action="store_true", help="Run small 4x4 or 8x8 quantum circuit demonstration.")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark comparison across test images.")

    args = parser.parse_args()

    if args.benchmark:
        print("Launching full benchmark suite...")
        import scripts.benchmark as bench
        bench.main()
        return

    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        print("\n=========================================================================")
        print("         RUNNING QUANTUM CIRCUIT SIMULATION DEMONSTRATION               ")
        print("=========================================================================")
        img = generate_synthetic_image(size=4, pattern='square')
        print("Input 4x4 Image Matrix:\n", img)

        runner = QSEDRunner(q_bits=8, mode='quantum')
        results = runner.run_pipeline(img)

        print("\nNEQR Circuit Qubits:", results['neqr_circuit'].num_qubits)
        print("NEQR Circuit Gate Count:", results['neqr_circuit'].count_ops())
        print("\nFinal Edge Detection Binary Matrix:\n", results['final_edges'])

        save_image(results['final_edges'] * 255, os.path.join(args.output, "demo_4x4_edges.png"))
        print(f"\nDemo edge image saved to: {os.path.join(args.output, 'demo_4x4_edges.png')}")

    elif args.image:
        print(f"\nLoading input image: {args.image}")
        img = load_image(args.image, target_size=(args.size, args.size))

        runner = QSEDRunner(q_bits=8, mode=args.mode)
        results = runner.run_pipeline(img)

        out_edge_path = os.path.join(args.output, "qsed_edges.png")
        save_image(results['final_edges'] * 255, out_edge_path)

        plot_path = os.path.join(args.output, "qsed_pipeline_breakdown.png")
        plot_qsed_pipeline_results(results, save_path=plot_path)

        print(f"Final quantum edge image saved to: {out_edge_path}")
        print(f"Pipeline visualization saved to: {plot_path}")

    else:
        print("No image provided. Running on default synthetic 64x64 test image...")
        img = generate_synthetic_image(size=64, pattern='checkerboard')

        runner = QSEDRunner(q_bits=8, mode=args.mode)
        results = runner.run_pipeline(img)

        out_edge_path = os.path.join(args.output, "synthetic_edges.png")
        save_image(results['final_edges'] * 255, out_edge_path)
        print(f"Synthetic test edge image saved to: {out_edge_path}")


if __name__ == "__main__":
    main()
