# admin.py
from core import *
import os
import torch
import math
from utils import build_optimizer
from dataclasses import dataclass
import torch.nn.functional as F
from datasets import load_dataset

# =========================================================
# CONFIG
# =========================================================
@dataclass
class GPTConfig:
    vocab_size: int = 4000
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    mlp_ratio: int = 4

    # performance
    dtype: torch.dtype = torch.bfloat16
    compile: bool = False
    use_rope: bool = True
    use_kv_cache: bool = True
    gradient_checkpointing: bool = True 
    use_flash_attention: bool = True 

    # activation
    activation: str = "swiglu"
    # relu | relu2 | gelu | silu | swiglu | geglu | reglu | mish | snake

    # optimization
    optimizer: str = "adamw"   # adamw | lion | sgd
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 32
    sequence_length: int = 64
    
    # advanced optimizations
    dropout: float = 0.1
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * accumulation_steps

    # SAM (Sharpness-Aware Minimization)
    sam: bool = False
    sam_rho: float = 0.05

    # init
    init_std: float = 0.02
    
    # training
    total_steps: int = 2500  # Total optimizer steps
    warmup_steps: int = 100
    val_interval: int = 25  # Validate every N steps
    use_ema: bool = True
    ema_decay: float = 0.99
    val_split: float = 0.05  # Hold 5% for validation

cfg = GPTConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use BF16 on GPU (Ampere+), fall back to float32 on CPU for stability
if device == "cpu":
    cfg.dtype = torch.float32
torch.set_float32_matmul_precision("high")

# Initialize tokenizer at module level for imports
tokenizer = None

# =========================================================
# GENERATION WITH SAMPLING
# =========================================================
@torch.no_grad()
def generate(model, idx, steps, temperature=0.9, top_k=40, use_cache=True):
    """Generate text with temperature and top-k sampling and KV cache."""
    model.eval()
    cache = None
    
    for _ in range(steps):
        if use_cache and cache is not None:
            # Use only last token with cache
            logits, cache = model(idx[:, -1:], cache)
        else:
            logits, cache = model(idx, cache)
        
        logits = logits[:, -1:] / temperature  # Keep batch dimension
        
        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits_mask = torch.full_like(logits, float('-inf'))
            logits_mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_mask
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


if __name__ == "__main__":
    # =========================================================
    # DEMO
    # =========================================================
    # Load dataset
    print("Loading dataset...")

    # Load exactly 10,000 examples from the train split
    ds = load_dataset("roneneldan/TinyStories", split="train[:10000]")

    # Extract the text field
    texts = [sample["text"] for sample in ds]

    # Join into a single string for tokenizer
    text = "\n".join(texts)

    print(f"Dataset loaded: {len(text)} characters")

    # Train tokenizer on Tiny Shakespeare
    print("\n" + "=" * 50)
    print("Training BPE Tokenizer...")
    print("=" * 50)
    tokenizer = BPETokenizer(cfg.vocab_size)
    tokenizer.train(text)
    print(f"Vocabulary size: {cfg.vocab_size}")
    print(f"Special token IDs: pad={tokenizer.pad_id}, unk={tokenizer.unk_id}, bos={tokenizer.bos_id}, eos={tokenizer.eos_id}")

    # Prepare training data with batching - EFFICIENT VERSION
    text_tokens = tokenizer.encode(text)

    # Create sequences directly in batch format (no intermediate reshaping)
    sequences = []
    targets = []

    for i in range(0, len(text_tokens) - cfg.sequence_length, cfg.sequence_length // 2):  # 50% overlap
        seq = text_tokens[i:i + cfg.sequence_length]
        if len(seq) == cfg.sequence_length:  # Only use full sequences
            tgt = text_tokens[i + 1:i + cfg.sequence_length + 1]
            sequences.append(seq)
            targets.append(tgt)

    # Convert to tensors once on device
    x_data = torch.tensor(sequences, device=device, dtype=torch.long)
    y_data = torch.tensor(targets, device=device, dtype=torch.long)

    print(f"Created {len(sequences)} sequences")
    print(f"Data shape: x={x_data.shape}, y={y_data.shape}")
    print(f"Total tokens: {len(sequences) * cfg.sequence_length}")

    # Train/val split
    val_count = int(x_data.shape[0] * cfg.val_split)
    val_idx = torch.randperm(x_data.shape[0])[:val_count]
    train_idx = torch.ones(x_data.shape[0], dtype=torch.bool)
    train_idx[val_idx] = False

    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]

    print(f"Train: {x_train.shape[0]} | Val: {x_val.shape[0]}")

    # Initialize model
    print("\nInitializing GPT model...")
    model = GPT(cfg).to(device=device, dtype=cfg.dtype)
    if cfg.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg)

    # EMA (Exponential Moving Average) for smooth weights
    ema_state_dict = None

    print("\n" + "=" * 50)
    print("Training Started...")
    print("=" * 50)
    model.train()

    # BF16 training (no scaler needed)
    best_val_loss = float('inf')
    ema_state_dict = None

    # Create infinite loader for continuous sampling
    def infinite_loader():
        while True:
            perm = torch.randperm(x_train.shape[0])
            for batch_start in range(0, x_train.shape[0], cfg.batch_size):
                batch_end = min(batch_start + cfg.batch_size, x_train.shape[0])
                idx = perm[batch_start:batch_end]
                yield x_train[idx], y_train[idx]

    loader = infinite_loader()

    # Training loop: fixed number of steps with gradient accumulation
    for step in range(cfg.total_steps):
        # Warmup learning rate
        if step < cfg.warmup_steps:
            lr_scale = step / cfg.warmup_steps
        else:
            # Cosine annealing from warmup to 0
            progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        base_opt = optimizer.base if cfg.sam else optimizer
        for param_group in base_opt.param_groups:
            param_group['lr'] = cfg.lr * lr_scale
        
        # Gradient accumulation loop
        accumulated_loss = 0.0
        for acc_step in range(cfg.gradient_accumulation_steps):
            x_batch, y_batch = next(loader)
            
            # Mixed precision with torch.autocast (BF16 on GPU, FP32 on CPU)
            with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                logits, _ = model(x_batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    y_batch.reshape(-1)
                )
            
            # Scale loss by accumulation steps
            scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        # Gradient clipping after accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # SAM: first step to perturb weights
        if cfg.sam:
            optimizer.first_step()
            
            # Recompute loss at perturbed weights (single batch for efficiency)
            x_batch, y_batch = next(loader)
            with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                logits, _ = model(x_batch)
                loss_perturbed = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    y_batch.reshape(-1)
                )
            loss_perturbed.backward()
            optimizer.second_step()
        else:
            base_opt.step()
            base_opt.zero_grad()
        
        # Average accumulated loss
        accumulated_loss /= cfg.gradient_accumulation_steps
        
        # Update EMA
        if cfg.use_ema:
            if ema_state_dict is None:
                ema_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    ema_state_dict[k] = cfg.ema_decay * ema_state_dict[k] + (1 - cfg.ema_decay) * v
        
        # Periodic validation
        if (step + 1) % cfg.val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_start in range(0, x_val.shape[0], cfg.batch_size):
                    batch_end = min(batch_start + cfg.batch_size, x_val.shape[0])
                    x_vbatch = x_val[batch_start:batch_end]
                    y_vbatch = y_val[batch_start:batch_end]
                    logits, _ = model(x_vbatch)
                    val_loss += F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        y_vbatch.reshape(-1)
                    ).item()
                    val_batches += 1
            
            val_loss /= max(val_batches, 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
                if cfg.use_ema:
                    torch.save(ema_state_dict, "best_ema_model.pt")
            
            model.train()
            print(f"Step {step+1:5d} | Train Loss {accumulated_loss:.4f} | Val Loss {val_loss:.4f}")

    # Load best EMA model for generation if available, otherwise best model
    print("\nLoading best model...")
    if cfg.use_ema and os.path.exists("best_ema_model.pt"):
        print("Using EMA model for generation")
        ema_state_dict = torch.load("best_ema_model.pt")
        model.load_state_dict(ema_state_dict)
    else:
        model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    print("\n" + "=" * 50)
    print("Text Generation")
    print("=" * 50)

    # Generate text with different prompts
    food_prompts = ["cooking", "fresh", "delicious", "baking"]

    for prompt in food_prompts:
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
        
        generated = generate(model, prompt_tensor, steps=20, temperature=0.9, top_k=30, use_cache=cfg.use_kv_cache)
        output_text = tokenizer.decode(generated[0].tolist())
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {output_text}\n")