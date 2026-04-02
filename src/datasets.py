from datasets import load_dataset
import numpy as np
from src.extraction.activations import ActivationExtractor


class DatasetBuilder:
    """
    Builds dataset of (activation, label) pairs for hallucination detection.
    label = 1 → hallucination
    label = 0 → correct
    """

    def __init__(self, extractor: ActivationExtractor):
        self.extractor = extractor

    def build_dataset(self, num_samples: int = 100, enhanced: bool = False):
        # Load TruthfulQA dataset
        dataset = load_dataset("truthful_qa", "generation")
        validation = dataset["validation"]

        print(f"Loaded {len(validation)} samples")
        if enhanced:
            print("Using ENHANCED features (multi-layer + confidence)")
        print("Using CONTRAST labeling (correct vs incorrect completions)")

        X = []  # activations
        y = []  # labels

        for i, example in enumerate(validation):
            if i >= num_samples:
                break

            try:
                question = example["question"]
                correct_answers = example["correct_answers"]
                incorrect_answers = example["incorrect_answers"]

                # For each CORRECT answer → label 0
                for ans in correct_answers[:2]:  # cap at 2 per question
                    prompt = f"Q: {question}\nA: {ans}"
                    if enhanced:
                        activation = self.extractor.get_enhanced_features(prompt)
                    else:
                        activation = self.extractor.get_activation(prompt)
                    X.append(activation)
                    y.append(0)

                # For each INCORRECT answer → label 1
                for ans in incorrect_answers[:2]:  # cap at 2 per question
                    prompt = f"Q: {question}\nA: {ans}"
                    if enhanced:
                        activation = self.extractor.get_enhanced_features(prompt)
                    else:
                        activation = self.extractor.get_activation(prompt)
                    X.append(activation)
                    y.append(1)

                # Progress logging
                if i % 10 == 0:
                    print(
                        f"Processed {i}/{num_samples} | "
                        f"Total pairs: {len(y)} | "
                        f"Hallucinations: {sum(y)}"
                    )

            except Exception as e:
                print(f"Skipping example {i} due to error: {e}")
                continue

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print("\nDataset built successfully!")
        print(f"Total samples: {len(y)}")
        print(f"Hallucinations: {y.sum()}")
        print(f"Correct: {len(y) - y.sum()}")

        return X, y