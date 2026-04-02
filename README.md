# TrueLens: Internal State Hallucination Detector

A lightweight, experimental probing framework that detects Large Language Model (LLM) hallucinations by examining their internal mathematical states during text generation.

Instead of relying on black-box text comparisons or logit entropy alone, TrueLens hooks directly into the core transformer architecture (using `transformer_lens` and PyTorch) to extract the **residual stream activations** from middle computing layers (e.g., Layer 22 out of 24). It then trains lightweight classifiers to identify the hidden semantic patterns of "truthfulness" versus "hallucination."

This project is loosely inspired by the concepts in the [INSIDE paper (ICLR 2024)](https://arxiv.org/abs/2402.03744) and [Azaria & Mitchell (2023)](https://arxiv.org/abs/2304.13734).

## 🚀 Features

* **Internal State Probing**: Extracts raw dense vectors (1024+ dimensions) directly from the `resid_post` transformer cache.
* **Multi-Layer Support**: Concatenates features from multiple layers (e.g., layers 16-23) to capture high-level semantic divergence.
* **Confidence Enhancements**: Appends mathematical confidence metrics directly from the output logits (Entropy, Top-1 probability, Top1-Top2 gap, Top-5 mass).
* **Multi-Model Support**: Native support for Pythia (160m-1B) and GPT-2 (small-large) models.
* **4-bit Quantization**: Uses `bitsandbytes` to load and probe 7B-parameter models (like DeepSeek Coder 6.7B) perfectly onto consumer laptops with 4GB VRAM.
* **Contrast Labeling**: Extracts high-quality training data from [TruthfulQA](https://huggingface.co/datasets/truthful_qa) by feeding the model both known correct and incorrect answer completions.
* **Fast Classifiers**: Evaluates features using balanced combinations of Logistic Regression, SVM (RBF kernel), deep MLPs, and LightGBM.
* **Real-time API**: Includes a lightning-fast `FastAPI` instance to query the trained probes.

## ⚙️ Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### 1. Benchmark Across Models
To test the probe on multiple local LLMs (auto-sizes to your available VRAM):
```bash
python benchmark.py --samples 200 --enhanced
```

Test a massive model (like DeepSeek Coder 6.7B) packed into 4GB VRAM using 4-bit mode:
```bash
python benchmark.py --models deepseek-7b --samples 100
```

### 2. Train Custom Probe
Train the classifiers using TruthfulQA contrast pairs on a target model (e.g., GPT-2 Medium):
```bash
python train.py --samples 800 --layer 22 --enhanced
```

### 3. Run the API
Serve the trained models on a local REST server:
```bash
uvicorn src.api.app:app --reload
```

## ⚠️ Limitations

* **Open-Weights Models Only**: This architecture cannot be applied to closed-source systems like OpenAI's `GPT-4` or Anthropic's `Claude`. Because those platforms are locked behind APIs, we cannot access the intermediate `resid_post` layers or computation graph required to run the feature extraction.
* **Heavy VRAM Taxation**: Holding the full computation graph to capture internal activations requires significant VRAM. Without 4-bit `bitsandbytes` quantization, anything over 1.5B parameters will easily overflow standard consumer 4GB/8GB GPUs.
* **Task Distribution**: The current probes are specialized on the TruthfulQA dataset. A probe trained specifically on TriviaQA or CoQA may be needed for different prompt formats.
* **Ollama Incompatible**: You cannot use `llama.cpp` or `ollama` as a backend since they are optimized, black-box text generators that do not expose intermediate PyTorch layer math.

## Project Structure
```text
hallucination-detector/
├── train.py                     # Entry point for training probes
├── benchmark.py                 # Benchmarking script for testing 7B+ models
├── requirements.txt             # Dependencies
├── models/                      # Pickled sklearn classifiers
├── src/
│   ├── api/app.py               # FastAPI backend
│   ├── datasets.py              # TruthfulQA contrast labeler
│   ├── extraction/
│   │   └── activations.py       # HookedTransformer 4-bit layer extractor
│   └── probing/
│       └── probe.py             # Feature scaler and model trainer
```
