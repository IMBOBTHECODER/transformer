import torch
from core import GPT, BPETokenizer
from admin import cfg, device, generate, HAS_XLA
from datasets import load_dataset
import os
import argparse

cfg.compile_mode = "max-autotune"

# Parse command-line arguments (including checkpoint selection)
parser = argparse.ArgumentParser(description="Generate text using the trained transformer model")
parser.add_argument("prompt", nargs="?", default=None, help="Text prompt to start generation")
parser.add_argument("steps", nargs="?", type=int, default=100, help="Number of tokens to generate (default: 100)")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8)")
parser.add_argument("--top_k", type=int, default=30, help="Top-k for sampling (default: 30)")
parser.add_argument("--checkpoint", type=str, default="ema", choices=["ema", "model"], 
                    help="Which checkpoint to load: 'ema' (best_ema_model.pt) or 'model' (best_model.pt) (default: ema)")

args = parser.parse_args()

# Initialize model with the same config
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

# Try to load the specified checkpoint
checkpoint_name = "best_ema_model.pt" if args.checkpoint == "ema" else "best_model.pt"
checkpoint_path = f"model/{checkpoint_name}"

if os.path.exists(checkpoint_path):
    try:
        if HAS_XLA:
            # On TPU: load to CPU first to strip XLA storage tags
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        else:
            # On GPU/CPU: load directly to device
            state_dict = torch.load(checkpoint_path, map_location=device)
        
        # If state dict has "_orig_mod." prefix (from torch.compile), remove it
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"Loaded {args.checkpoint.upper()} checkpoint from {checkpoint_path}")
    except RuntimeError as e:
        print(f"Warning: Could not load checkpoint (architecture mismatch): {e}")
        print("Using randomly initialized model instead")
else:
    print(f"No {args.checkpoint} checkpoint found at {checkpoint_path} - using randomly initialized model")

# Initialize tokenizer - load if exists, otherwise train and save
tokenizer = BPETokenizer(cfg.vocab_size)
tokenizer_path = f"tokenizer_{cfg.tokenizer_id}.json"

if os.path.exists(tokenizer_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer.load(tokenizer_path)
else:
    print("Loading dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:100000]")
    texts = [sample["text"] for sample in ds]
    text = "\n".join(texts)
    
    print("Training tokenizer...")
    tokenizer.train(text)
    tokenizer.save(tokenizer_path)

# Use command-line arguments or prompt for input
if args.prompt is None:
    print("Usage: python test.py \"Your prompt here\" [steps] [--checkpoint {ema,model}]")
    print("Example: python test.py \"Once upon a time\" 100 --checkpoint model")
    print()
    prompt = input("Enter a prompt: ").strip()
else:
    prompt = args.prompt

steps = args.steps
temperature = args.temperature
top_k = args.top_k

# Now use the model for inference
with torch.no_grad():
    print(f"\nGenerating {steps} tokens from prompt: '{prompt}'")
    print("-" * 60)
    
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    # Allocate KV cache for faster generation (O(n) instead of O(n²) complexity)
    # Cache is preallocated for prompt + generated tokens
    max_seq_len = len(prompt_tokens) + steps
    cache = model.allocate_kv_cache(
        batch_size=1,
        max_seq_len=max_seq_len,
        device=device,
        dtype=cfg.dtype
    )
    
    # Prefill cache with prompt tokens (uses cache but only for precomputation)
    with torch.no_grad():
        _, cache = model(prompt_tensor, cache=cache)
    
    # Generate new tokens with cached KV (single token per step)
    generated = generate(model, prompt_tensor, steps=steps, temperature=temperature, 
                        top_k=top_k, use_cache=True)
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)