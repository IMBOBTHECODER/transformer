"""GPU-optimized text generation with KV cache for efficient inference."""

import torch
import os
import argparse
from core import GPT, BPETokenizer
from admin import cfg, device, generate
from datasets import load_dataset

# Enable max autotune for best GPU performance
cfg.compile_mode = "max-autotune"

# =====================================================
# ARGUMENT PARSING
# =====================================================
parser = argparse.ArgumentParser(description="Generate text using trained transformer")
parser.add_argument("prompt", nargs="?", default=None, help="Starting prompt")
parser.add_argument("steps", nargs="?", type=int, default=100, help="Tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling")
parser.add_argument("--checkpoint", type=str, default="ema", choices=["ema", "model"],
                    help="Checkpoint: ema (default) or model")
args = parser.parse_args()

# =====================================================
# LOAD MODEL CHECKPOINT
# =====================================================
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

checkpoint_name = "best_ema_model.pt" if args.checkpoint == "ema" else "best_model.pt"
checkpoint_path = f"model/{checkpoint_name}"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Remove torch.compile wrapper prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(state_dict)
        print(f"✓ Loaded {args.checkpoint.upper()} checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        print(f"✗ Architecture mismatch - {e}")
        print("Using randomly initialized model")
else:
    print(f"✗ Checkpoint not found at {checkpoint_path}")
    print("Using randomly initialized model")

# =====================================================
# LOAD OR TRAIN TOKENIZER
# =====================================================
tokenizer = BPETokenizer(cfg.vocab_size)
tokenizer_path = f"tokenizer_{cfg.tokenizer_id}.json"

if os.path.exists(tokenizer_path):
    tokenizer.load(tokenizer_path)
else:
    print("Training tokenizer on TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:100000]")
    text = "\n".join([sample["text"] for sample in ds])
    tokenizer.train(text)
    tokenizer.save(tokenizer_path)

# =====================================================
# PREPARE GENERATION PARAMETERS
# =====================================================
if args.prompt is None:
    print("\nUsage: python test.py \"Your prompt\" [steps] [--checkpoint {ema,model}]")
    print("Example: python test.py \"Once upon a time\" 100 --checkpoint ema")
    print()
    prompt = input("Enter prompt: ").strip()
else:
    prompt = args.prompt

steps = args.steps
temperature = args.temperature
top_k = args.top_k

# =====================================================
# GENERATE TEXT WITH KV CACHE
# =====================================================
print(f"\nGenerating {steps} tokens...")
print("-" * 60)

model.eval()
with torch.no_grad():
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    # Preallocate KV cache: single pass through prompt, then O(n) generation (vs O(n²))
    max_seq_len = len(prompt_tokens) + steps
    cache = model.allocate_kv_cache(batch_size=1, max_seq_len=max_seq_len, 
                                    device=device, dtype=cfg.dtype)
    
    # Prefill: process prompt through model to populate cache
    _, cache = model(prompt_tensor, cache=cache)
    
    # Generate tokens using cached KV (one token per forward pass)
    generated = generate(model, prompt_tensor, steps=steps, temperature=temperature,
                        top_k=top_k, use_cache=True)
    
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)