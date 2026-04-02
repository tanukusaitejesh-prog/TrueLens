"""
Multi-model benchmark for hallucination probing.
Trains & evaluates probes across multiple LLMs supported by TransformerLens.

Usage:
    python benchmark.py                     # all models that fit your GPU
    python benchmark.py --models gpt2-small gpt2-medium
    python benchmark.py --samples 200       # fewer samples for faster runs
"""

import argparse
import os
import json
import time
import torch
import numpy as np
import gc
from datetime import datetime

# Model configs: (name, TransformerLens ID, recommended probe layer, approx VRAM GB)
MODEL_CONFIGS = {
    "gpt2-small":   {"tl_name": "gpt2-small",   "n_layers": 12, "probe_layer": 10, "vram_gb": 0.5, "quant": False},
    "gpt2-medium":  {"tl_name": "gpt2-medium",  "n_layers": 24, "probe_layer": 22, "vram_gb": 1.5, "quant": False},
    "gpt2-large":   {"tl_name": "gpt2-large",   "n_layers": 36, "probe_layer": 34, "vram_gb": 3.2, "quant": False},
    "pythia-160m":  {"tl_name": "pythia-160m",  "n_layers": 12, "probe_layer": 10, "vram_gb": 0.6, "quant": False},
    "pythia-410m":  {"tl_name": "pythia-410m",  "n_layers": 24, "probe_layer": 22, "vram_gb": 1.6, "quant": False},
    "pythia-1b":    {"tl_name": "pythia-1b",    "n_layers": 16, "probe_layer": 14, "vram_gb": 3.8, "quant": False},
    "deepseek-7b":  {"tl_name": "deepseek-ai/deepseek-coder-6.7b-base", "n_layers": 32, "probe_layer": 28, "vram_gb": 4.5, "quant": True},
}


def clear_gpu():
    """Free GPU memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_benchmark(model_key: str, config: dict, num_samples: int, enhanced: bool):
    """Train and evaluate probe for a single model. Returns results dict."""
    from src.extraction.activations import ActivationExtractor
    from src.datasets import DatasetBuilder
    from src.probing.probe import HallucinationProbe

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_key}")
    print(f"  Layers: {config['n_layers']} | Probe layer: {config['probe_layer']} | ~{config['vram_gb']}GB VRAM")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 1. Initialize extractor directly (not through probe, to control model name)
        extractor = ActivationExtractor(
            model_name=config["tl_name"],
            layer=config["probe_layer"],
            quantize_4bit=config.get("quant", False)
        )

        # 2. Build dataset
        builder = DatasetBuilder(extractor=extractor)
        X, y = builder.build_dataset(num_samples=num_samples, enhanced=enhanced)

        # 3. Train probes
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        import lightgbm
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, roc_auc_score

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        classifiers = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
            "MLP": MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, early_stopping=True),
            "lightgbm": lightgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, is_unbalance=True, verbose=-1),
        }

        model_results = {}
        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            model_results[clf_name] = {"f1": round(f1, 4), "roc_auc": round(auc, 4)}
            print(f"  {clf_name:25s} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        elapsed = time.time() - start_time

        # Cleanup model from GPU
        del extractor
        del builder
        clear_gpu()

        return {
            "model": model_key,
            "n_layers": config["n_layers"],
            "probe_layer": config["probe_layer"],
            "num_samples": num_samples,
            "dataset_size": len(y),
            "hallucination_ratio": round(float(y.sum()) / len(y), 3),
            "feature_dim": X.shape[1],
            "enhanced": enhanced,
            "results": model_results,
            "time_seconds": round(elapsed, 1),
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        clear_gpu()
        return {
            "model": model_key,
            "error": str(e),
            "time_seconds": round(time.time() - start_time, 1),
        }


def print_summary_table(all_results):
    """Print a comparison table across all models."""
    print("\n")
    print("=" * 90)
    print("  BENCHMARK SUMMARY — Hallucination Probe Across LLMs")
    print("=" * 90)

    # Header
    print(f"\n  {'Model':<16} {'Layers':<8} {'Probe':<7} {'Feat':<7} | {'Best F1':<10} {'Best AUC':<10} {'Best Clf':<20} | {'Time':<8}")
    print("  " + "-" * 86)

    for r in all_results:
        if "error" in r:
            print(f"  {r['model']:<16} {'ERROR':>50} | {r.get('time_seconds',0):.0f}s")
            continue

        # Find best classifier by AUC
        best_clf = max(r["results"], key=lambda k: r["results"][k]["roc_auc"])
        best_f1 = r["results"][best_clf]["f1"]
        best_auc = r["results"][best_clf]["roc_auc"]

        print(
            f"  {r['model']:<16} {r['n_layers']:<8} L{r['probe_layer']:<5} {r['feature_dim']:<7} | "
            f"{best_f1:<10.4f} {best_auc:<10.4f} {best_clf:<20} | {r['time_seconds']:.0f}s"
        )

    # Detailed per-classifier breakdown
    print(f"\n\n  Detailed Results (ROC-AUC by model × classifier):")
    print(f"  {'Model':<16} | {'LR':<8} {'SVM':<8} {'MLP':<8} {'LGBM':<8}")
    print("  " + "-" * 55)

    for r in all_results:
        if "error" in r:
            continue
        res = r["results"]
        print(
            f"  {r['model']:<16} | "
            f"{res.get('logistic_regression',{}).get('roc_auc','N/A'):<8} "
            f"{res.get('SVM',{}).get('roc_auc','N/A'):<8} "
            f"{res.get('MLP',{}).get('roc_auc','N/A'):<8} "
            f"{res.get('lightgbm',{}).get('roc_auc','N/A'):<8}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark hallucination probes across LLMs")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to test. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--samples", type=int, default=200,
                        help="TruthfulQA samples per model (default: 200)")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use multi-layer + confidence features")
    args = parser.parse_args()

    # Select models
    if args.models:
        selected = {k: MODEL_CONFIGS[k] for k in args.models if k in MODEL_CONFIGS}
    else:
        # Auto-select based on available VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"Detected GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
            selected = {k: v for k, v in MODEL_CONFIGS.items() if v["vram_gb"] < vram_gb * 0.85}
        else:
            selected = {"gpt2-small": MODEL_CONFIGS["gpt2-small"]}

    print(f"\nModels to benchmark: {list(selected.keys())}")
    print(f"Samples per model: {args.samples}")
    print(f"Enhanced features: {args.enhanced}")

    # Run benchmarks
    all_results = []
    for model_key, config in selected.items():
        result = run_benchmark(model_key, config, args.samples, args.enhanced)
        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/benchmark_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
