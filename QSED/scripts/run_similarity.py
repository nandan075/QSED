import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classical.image_loader import load_image
from src.feature.similarity import compute_similarity, save_similarity_outputs


def main():
    parser = argparse.ArgumentParser(
        description="QSED Quantum Entropy-Based Image Similarity Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_similarity.py --image1 img1.jpg --image2 img2.jpg --size 1024
  python scripts/run_similarity.py --image1 img1.jpg --image2 img2.jpg --size 512 --output outputs/sim/
        """
    )
    parser.add_argument('--image1', type=str, required=True, help="Path to first image")
    parser.add_argument('--image2', type=str, required=True, help="Path to second image")
    parser.add_argument('--size', type=int, default=1024, help="Image size (power of 2, default: 1024)")
    parser.add_argument('--neighborhood', type=int, default=3, help="Sliding window size (default: 3)")
    parser.add_argument('--output', type=str, default='outputs/similarity/', help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("     QSED Quantum Entropy Image Similarity Comparison")
    print("=" * 65)
    print()
    
    # 1. Load images
    print(f"Loading images at {args.size}x{args.size}...")
    t0 = time.time()
    image_a = load_image(args.image1, target_size=(args.size, args.size))
    image_b = load_image(args.image2, target_size=(args.size, args.size))
    print(f"  Loaded in {time.time() - t0:.2f}s")
    print()
    
    # 2. Compute similarity
    print(f"Computing 3x3 Von Neumann Entropy maps...")
    print(f"  Window positions: {(args.size - args.neighborhood + 1) ** 2:,}")
    t1 = time.time()
    results = compute_similarity(
        image_a, image_b,
        neighborhood_size=args.neighborhood
    )
    print(f"  Completed in {time.time() - t1:.2f}s")
    print()
    
    # 3. Display results
    print("-" * 65)
    print(f"  Structural Complexity Match:     {results['complexity_similarity']:6.2f}%")
    print(f"  Spatial Layout (Pyramid) Match:  {results['spatial_layout_similarity']:6.2f}%")
    print(f"  Pooled Spatial Correlation:      {results['pooled_correlation_similarity']:6.2f}%")
    print("-" * 65)
    print()
    print(f"  [*] Combined Similarity Score:   {results['combined_similarity']:6.2f}%")
    print()
    
    # 4. Save outputs
    print(f"Saving outputs to {args.output}...")
    save_similarity_outputs(results, image_a, image_b, args.output)
    print("  [OK] similarity_report.txt")
    print("  [OK] entropy_diff_map.csv")
    print("  [OK] similarity_comparison.png")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
