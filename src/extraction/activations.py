import torch
import numpy as np
import transformer_lens as t
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
class ActivationExtractor:
    def __init__(self, model_name: str = "gpt2-medium", layer: int = 22, device: Optional[str] = None, quantize_4bit: bool = False):
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}")

        if quantize_4bit:
            print("  [INFO] Using 4-bit quantization (bitsandbytes) to save VRAM...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # When wrapping quantized models in TransformerLens, we must disable weight modifications
            self.model = t.HookedTransformer.from_pretrained(
                self.model_name,
                hf_model=hf_model,
                tokenizer=tokenizer,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False
            )
        else:
            self.model = t.HookedTransformer.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded. Extracting from layer {self.layer}")
    def get_activation(self,prompt: str)->np.ndarray:
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
        activation = cache["resid_post", self.layer][0, -1, :]
        return activation.cpu().numpy()
    def get_attention_patterns(self,prompt:str)->dict:
        tokens = self.model.to_tokens(prompt)
        str_tokens = self.model.to_str_tokens(prompt)
        with torch.no_grad():
            s, cache = self.model.run_with_cache(tokens)
        patterns={}
        for layer in range(self.model.cfg.n_layers):
            attn = cache["pattern",layer]
            patterns[layer]=attn[0].cpu().numpy().tolist()
        return {"tokens": str_tokens,"patterns":patterns}
    def get_multi_layer_activation(self, prompt: str, layers: list = None) -> np.ndarray:
        """Concatenate residual stream from multiple layers for richer features."""
        if layers is None:
            layers = list(range(16, self.model.cfg.n_layers))  # layers 16-23
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
        acts = []
        for layer in layers:
            activation = cache["resid_post", layer][0, -1, :]
            acts.append(activation.cpu().numpy())
        return np.concatenate(acts)

    def get_confidence_features(self, prompt: str) -> np.ndarray:
        """Extract entropy, top-1 prob, and top1-top2 gap from logit distribution."""
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            logits = self.model(tokens)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        # Entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        # Top-1 probability
        top_probs, _ = torch.topk(probs, 5)
        top1_prob = top_probs[0].item()
        # Gap between top-1 and top-2
        top1_top2_gap = (top_probs[0] - top_probs[1]).item()
        # Top-5 probability mass
        top5_mass = top_probs.sum().item()
        return np.array([entropy, top1_prob, top1_top2_gap, top5_mass])

    def get_enhanced_features(self, prompt: str, layers: list = None) -> np.ndarray:
        """Combine multi-layer activations + confidence features into one vector."""
        multi_act = self.get_multi_layer_activation(prompt, layers)
        conf = self.get_confidence_features(prompt)
        return np.concatenate([multi_act, conf])

    def get_top_prediction(self, prompt: str) -> str:
        tokens = self.model.to_tokens(prompt)
        with torch.no_grad():
            logits = self.model(tokens)
        top1_idx = logits[0, -1, :].argmax().item()
        return self.model.tokenizer.decode([top1_idx])

