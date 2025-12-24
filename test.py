import torch
from core import GPT, BPETokenizer
from admin import cfg, device, generate
from datasets import load_dataset
import os
import argparse

cfg.compile_mode = "max-autotune"

# Initialize model with the same config
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

# Load the EMA model weights - handle compiled model state dict
ema_state_dict = torch.load("model/best_ema_model.pt", map_location=device)

# If state dict has "_orig_mod." prefix (from torch.compile), remove it
if any(k.startswith("_orig_mod.") for k in ema_state_dict.keys()):
    ema_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ema_state_dict.items()}

model.load_state_dict(ema_state_dict)
model.eval()

# Initialize tokenizer - load if exists, otherwise train and save
tokenizer = BPETokenizer(cfg.vocab_size)
tokenizer_path = "tokenizer.json"

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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate text using the trained transformer model")
parser.add_argument("prompt", nargs="?", default=None, help="Text prompt to start generation")
parser.add_argument("steps", nargs="?", type=int, default=100, help="Number of tokens to generate (default: 100)")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8)")
parser.add_argument("--top_k", type=int, default=30, help="Top-k for sampling (default: 30)")

args = parser.parse_args()

# Use command-line arguments or prompt for input
if args.prompt is None:
    print("Usage: python test.py \"Your prompt here\" [steps]")
    print("Example: python test.py \"Once upon a time\" 100")
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
    
    # Generate new tokens
    generated = generate(model, prompt_tensor, steps=steps, temperature=temperature, top_k=top_k)
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)