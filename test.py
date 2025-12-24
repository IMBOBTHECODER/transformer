import torch
from core import GPT, BPETokenizer
from admin import cfg, device, generate
from datasets import load_dataset
import os

# Initialize model with the same config
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

# Load the EMA model weights - handle compiled model state dict
ema_state_dict = torch.load("best_ema_model.pt", map_location=device)

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

# Now use the model for inference
with torch.no_grad():
    # Example: generate text from a prompt
    prompt = "Once upon a time"  
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    # Generate 50 new tokens
    generated = generate(model, prompt_tensor, steps=100, temperature=0.8, top_k=30)
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)