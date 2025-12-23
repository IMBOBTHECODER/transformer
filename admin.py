# admin.py
from core import *
import os
import torch
import math
from utils import build_optimizer
from dataclasses import dataclass
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
@dataclass
class GPTConfig:
    vocab_size: int = 16000
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 8
    mlp_ratio: int = 4

    # performance
    dtype: torch.dtype = torch.float32
    compile: bool = True
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
    batch_size: int = 64
    sequence_length: int = 512
    
    # advanced optimizations
    dropout: float = 0.1
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * accumulation_steps

    # SAM (Sharpness-Aware Minimization)
    sam: bool = False
    sam_rho: float = 0.05

    # init
    init_std: float = 0.02
    
    # training
    total_steps: int = 2000  # Total optimizer steps
    warmup_steps: int = 200
    val_interval: int = 100  # Validate every N steps
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
# PYTORCH DATASET AND DATALOADER
# =========================================================
class TokenSequenceDataset(Dataset):
    """Efficient dataset for token sequences with minimal memory footprint."""
    
    def __init__(self, token_stream, seq_len, stride=None):
        """
        Args:
            token_stream: List/array of token IDs
            seq_len: Sequence length for each sample
            stride: Stride for sliding window (default: seq_len // 2 for 50% overlap)
        """
        self.token_stream = token_stream
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2
        
        # Calculate number of sequences we can extract
        self.num_sequences = max(1, (len(token_stream) - seq_len) // self.stride)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence at index without materializing entire dataset."""
        token_start = idx * self.stride
        token_end = token_start + self.seq_len
        
        # Ensure we don't go out of bounds
        if token_end >= len(self.token_stream):
            token_end = len(self.token_stream) - 1
            token_start = max(0, token_end - self.seq_len)
        
        seq = self.token_stream[token_start:token_end]
        tgt = self.token_stream[token_start + 1:token_end + 1]
        
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def create_data_loaders(token_stream, cfg, val_split=0.05, num_workers=0):
    """
    Create train and validation dataloaders with efficient batching.
    
    Args:
        token_stream: Complete token sequence
        cfg: Config object with batch_size and sequence_length
        val_split: Fraction of data for validation
        num_workers: Number of worker processes (0 for CPU, increase for GPU)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    total_tokens = len(token_stream)
    val_token_count = int(total_tokens * val_split)
    
    # Split data without materializing
    train_tokens = token_stream[val_token_count:]
    val_tokens = token_stream[:val_token_count]
    
    # Create datasets
    train_dataset = TokenSequenceDataset(train_tokens, cfg.sequence_length)
    val_dataset = TokenSequenceDataset(val_tokens, cfg.sequence_length)
    
    # Create dataloaders with pin_memory for GPU transfer efficiency
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),  # Speed up GPU transfer
        drop_last=True,  # Drop incomplete last batch
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=False,
    )
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")
    print(f"DataLoaders created with batch_size={cfg.batch_size}, num_workers={num_workers}")
    
    return train_loader, val_loader


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
    ds = load_dataset("roneneldan/TinyStories", split="train[:100000]")

    # Extract the text field
    texts = [sample["text"] for sample in ds]

    # Join into a single string for tokenizer
    text = "\n".join(texts)

    print(f"Dataset loaded: {len(text)} characters")

    # Train tokenizer on Tiny Shakespeare
    print("\n" + "=" * 50)
    print("Training BPE Tokenizer...")
    print("=" * 50)
    tokenizer = BPETokenizer(cfg.vocab_size, min_frequency=2)
    tokenizer.train(text, show_stats=True)
    
    print(f"Actual vocabulary size: {tokenizer.get_vocab_size()}")

    # Prepare training data - STREAMING BATCHES (no materialization)
    text_tokens = tokenizer.encode(text)
    total_tokens = len(text_tokens)
    compression_ratio = tokenizer.get_compression_ratio(text)
    
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio: {compression_ratio:.2f}x (chars/tokens)")
    
    # Create PyTorch DataLoaders for efficient batching and memory management
    train_loader, val_loader = create_data_loaders(
        text_tokens, 
        cfg, 
        val_split=cfg.val_split,
        num_workers=0  # Keep at 0 to avoid multiprocessing conflicts with tokenizers
    )

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

    # Training loop: fixed number of steps with gradient accumulation
    step = 0
    while step < cfg.total_steps:
        # Cycle through training data
        for x_batch, y_batch in train_loader:
            if step >= cfg.total_steps:
                break
            
            # Move to device if not already there
            x_batch = x_batch.to(device, dtype=torch.long)
            y_batch = y_batch.to(device, dtype=torch.long)
            
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
                
                # Get next batch for subsequent accumulation steps if needed
                if acc_step < cfg.gradient_accumulation_steps - 1:
                    try:
                        x_batch, y_batch = next(iter(train_loader))
                        x_batch = x_batch.to(device, dtype=torch.long)
                        y_batch = y_batch.to(device, dtype=torch.long)
                    except StopIteration:
                        # If we run out of data, just use current batch
                        pass
            
            # Gradient clipping after accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # SAM: first step to perturb weights
            if cfg.sam:
                optimizer.first_step()
                
                # Recompute loss at perturbed weights (single batch for efficiency)
                x_batch_sam, y_batch_sam = next(iter(train_loader))
                x_batch_sam = x_batch_sam.to(device, dtype=torch.long)
                y_batch_sam = y_batch_sam.to(device, dtype=torch.long)
                with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                    logits, _ = model(x_batch_sam)
                    loss_perturbed = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        y_batch_sam.reshape(-1)
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
                    # Use DataLoader for validation (more efficient)
                    for x_vbatch, y_vbatch in val_loader:
                        x_vbatch = x_vbatch.to(device, dtype=torch.long)
                        y_vbatch = y_vbatch.to(device, dtype=torch.long)
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
            
            step += 1