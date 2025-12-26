# test.py (GPU)
"""GPU-optimized text generation with KV cache."""

import torch
import os
import argparse
import torch.nn.functional as F
from core import GPT, BPETokenizer, compile_model
from admin import cfg, device, generate


# =========================================================
# ARGUMENT PARSING
# =========================================================
parser = argparse.ArgumentParser(description="Generate text using trained GPT model")
parser.add_argument("prompt", nargs="?", default=None, help="Text prompt")
parser.add_argument("steps", nargs="?", type=int, default=100, help="Tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling")
parser.add_argument("--checkpoint", type=str, default="ema", choices=["ema", "model"],
                    help="Checkpoint: 'ema' or 'model'")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


# =========================================================
# LOAD MODEL
# =========================================================
print("Initializing model...")
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

# Compile model for faster inference
if cfg.compile:
    cfg.compile_mode = "max-autotune"
    model = compile_model(model, cfg.compile_mode)
    print("✓ Model compiled (max-autotune)")

print(f"Model: {cfg.id} | Vocab: {cfg.vocab_size:,} | Device: {device}")

checkpoint_name = f"best_ema_model_{cfg.id}.pt" if args.checkpoint == "ema" else f"best_model_{cfg.id}.pt"
checkpoint_path = f"model/{checkpoint_name}"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Remove torch.compile wrapper prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict)
        print(f"✓ Loaded {args.checkpoint.upper()} checkpoint")
    except RuntimeError as e:
        print(f"✗ Checkpoint mismatch: {e}")
        print("  Using randomly initialized model")
else:
    print(f"⚠ No checkpoint at {checkpoint_path}")
    print("  Using randomly initialized model")


# =========================================================
# LOAD TOKENIZER
# =========================================================
print("Loading tokenizer...")
tokenizer = BPETokenizer(cfg.vocab_size, tokenizer_id=cfg.id)
tokenizer_path = f"tokenizer/tokenizer_{cfg.id}.json"

if os.path.exists(tokenizer_path):
    tokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded")
else:
    print(f"✗ Tokenizer not found at {tokenizer_path}")
    print("  Run: python admin.py to prepare tokenizer first")
    exit(1)


# =========================================================
# PREPARE GENERATION
# =========================================================
if args.prompt is None:
    print("\nUsage: python test.py \"prompt\" [steps] [--checkpoint {ema,model}] [--seed SEED]")
    print("Example: python test.py \"Once upon a time\" 100 --checkpoint ema --seed 42")
    print()
    prompt = input("Prompt: ").strip()
else:
    prompt = args.prompt

# Validate inputs
if not prompt:
    raise ValueError("Prompt cannot be empty")
if args.steps <= 0:
    raise ValueError("Steps must be positive")
if args.temperature <= 0:
    raise ValueError("Temperature must be positive")
if args.top_k < 0:
    raise ValueError("Top-k must be non-negative")

steps = args.steps
temperature = args.temperature
top_k = args.top_k

print(f"\nGenerating {steps} tokens from: '{prompt}'")
if args.seed is not None:
    print(f"Seed: {args.seed}")
print("-" * 60)


# =========================================================
# GENERATE TEXT
# =========================================================
model.eval()
with torch.no_grad():
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)

    # Generate with KV cache
    generated = generate(model, prompt_tensor, steps=steps, temperature=temperature,
                         top_k=top_k, use_cache=True)

    output_text = tokenizer.decode(generated[0].tolist())
    print("\nGenerated text:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)
