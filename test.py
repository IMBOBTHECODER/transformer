import torch
from core import GPT, BPETokenizer
from admin import cfg, device, generate
from datasets import load_dataset

# Initialize model with the same config
model = GPT(cfg).to(device=device, dtype=cfg.dtype)

# Load the EMA model weights
ema_state_dict = torch.load("best_ema_model.pt", map_location=device)
model.load_state_dict(ema_state_dict)
model.eval()

# Initialize and train tokenizer (or load if needed)
print("Loading dataset...")
ds = load_dataset("roneneldan/TinyStories", split="train[:10000]")
texts = [sample["text"] for sample in ds]
text = "\n".join(texts)

print("Training tokenizer...")
tokenizer = BPETokenizer(cfg.vocab_size)
tokenizer.train(text)

# Now use the model for inference
with torch.no_grad():
    # Example: generate text from a prompt
    prompt = "She touched me and said"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    # Generate 50 new tokens
    generated = generate(model, prompt_tensor, steps=100, temperature=0.8, top_k=30)
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)