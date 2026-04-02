"""
Training script for the Hallucination Probe.
Extracts activations from GPT-2 Medium on TruthfulQA,
then trains LR / SVM / MLP / LightGBM classifiers.

Usage:
    python train.py                          # default 100 samples, basic features
    python train.py --samples 817 --enhanced # all samples, multi-layer + confidence
"""

import argparse
import os
from src.datasets import DatasetBuilder
from src.probing.probe import HallucinationProbe

def main():
    parser = argparse.ArgumentParser(description="Train hallucination probes")
    parser.add_argument("--samples", type=int, default=100, help="Number of TruthfulQA samples to use")
    parser.add_argument("--layer", type=int, default=22, help="Transformer layer to extract from")
    parser.add_argument("--output", type=str, default="models/", help="Directory to save trained models")
    parser.add_argument("--enhanced", action="store_true", help="Use multi-layer activations + confidence features")
    args = parser.parse_args()

    # Ensure output dir ends with /
    output_dir = args.output if args.output.endswith("/") else args.output + "/"

    # 1. Initialize probe (loads GPT-2 model)
    print("=" * 50)
    print("Step 1: Initializing probe")
    print("=" * 50)
    probe = HallucinationProbe(layer=args.layer)

    # 2. Build dataset from TruthfulQA
    print("\n" + "=" * 50)
    print("Step 2: Building dataset from TruthfulQA")
    if args.enhanced:
        print("  Mode: ENHANCED (8 layers × 1024 + 4 confidence = 8196 features)")
    else:
        print(f"  Mode: BASIC (layer {args.layer} only = 1024 features)")
    print("=" * 50)
    builder = DatasetBuilder(extractor=probe.extractor)
    X, y = builder.build_dataset(num_samples=args.samples, enhanced=args.enhanced)
    print(f"Feature vector size: {X.shape[1]}")

    # 3. Train all classifiers
    print("\n" + "=" * 50)
    print("Step 3: Training classifiers")
    print("=" * 50)
    probe.fit(X, y)

    # 4. Print results
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    for name, metrics in probe.results.items():
        print(f"  {name:25s} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    # 5. Save models
    print("\n" + "=" * 50)
    print(f"Step 4: Saving models to {output_dir}")
    print("=" * 50)
    os.makedirs(output_dir, exist_ok=True)
    probe.save(path=output_dir)
    print("Done! Models saved.")

if __name__ == "__main__":
    main()
