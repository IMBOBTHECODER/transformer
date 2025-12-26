# test.py (TPU)

import torch
import os
import argparse
import torch.nn.functional as F
from core import GPT, BPETokenizer
from admin import cfg, device

# Use max-autotune for inference
cfg.compile_mode = "max-autotune"

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
args = parser.parse_args()

# =========================================================
# LOAD MODEL
# =========================================================
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

checkpoint_name = "best_ema_model.pt" if args.checkpoint == "ema" else "best_model.pt"
checkpoint_path = f"model/{checkpoint_name}"

if os.path.exists(checkpoint_path):
    try:
        # Load to CPU first for XLA compatibility
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Remove torch.compile wrapper prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"✓ Loaded {args.checkpoint.upper()} checkpoint")
    except RuntimeError as e:
        print(f"✗ Checkpoint mismatch: {e}")
        print("  Using randomly initialized model")
else:
    print(f"✗ No checkpoint at {checkpoint_path}")
    print("  Using randomly initialized model")

# =========================================================
# TOKENIZER
# =========================================================
tokenizer = BPETokenizer(cfg.vocab_size)
tokenizer_path = f"tokenizer_{cfg.id}.json"

if os.path.exists(tokenizer_path):
    print(f"✓ Tokenizer loaded")
    tokenizer.load(tokenizer_path)
else:
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run: python tpu/tokenizer.py")

# =========================================================
# GENERATION
# =========================================================
if args.prompt is None:
    print("\nUsage: python test.py \"prompt\" [steps] [--checkpoint {ema,model}]")
    print("Example: python test.py \"Once upon a time\" 100 --checkpoint ema")
    print()
    prompt = input("Prompt: ").strip()
else:
    prompt = args.prompt

steps = args.steps
temperature = args.temperature
top_k = args.top_k

print(f"\nGenerating {steps} tokens from: '{prompt}'")
print("-" * 60)

with torch.no_grad():
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    # Preallocate KV cache for efficient generation
    max_seq_len = len(prompt_tokens) + steps
    cache = model.allocate_kv_cache(
        batch_size=1,
        max_seq_len=max_seq_len,
        device=device,
        dtype=cfg.dtype
    )
    
    # Prefill phase: process entire prompt (variable T) in single compiled graph
    _, cache = model.forward_prefill(prompt_tensor, cache)
    
    # Decode phase: single-token generation (T=1, static shape for XLA)
    generated = prompt_tensor
    for _ in range(steps):
        # Single-token forward pass (T=1, static shape for XLA)
        logits, cache = model.forward_generate(generated[:, -1:], cache, step=_)
        logits = logits[:, -1:] / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits_mask = torch.full_like(logits, float('-inf'))
            logits_mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_mask
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(-1), 1).unsqueeze(1)
        generated = torch.cat([generated, next_token], dim=1)
    
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)