# admin.py
from core import *
import os
import time
import torch
import math
from utils import build_optimizer
from dataclasses import dataclass
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Try to import XLA for TPU support
try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    MpDeviceLoader = None

# =========================================================
# CONFIG
# =========================================================
@dataclass
class GPTConfig:
    # main
    vocab_size: int = 16000
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    mlp_ratio: int = 4
    yarn_scale: float = 1.0  # YaRN: scale for RoPE max_seq_len during generation
    # n_kv_heads: n_heads // 2

    # performance
    dtype: torch.dtype = torch.float32   # bf16 | float32 | float16
    compile: bool = True
    compile_mode: str = "reduce-overhead"  # Training: reduce-overhead (fast startup) | Inference: max-autotune (max perf)
    use_rope: bool = True
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
    batch_size: int = 128
    sequence_length: int = 512
    
    # advanced optimizations
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1  # TPU: disable accumulation (use batch_size instead for XLA fusion)

    # SAM (Sharpness-Aware Minimization)
    sam: bool = False
    sam_rho: float = 0.08

    # init
    init_std: float = 0.02
    
    # training
    total_steps: int = 2000
    val_interval: int = 200
    use_ema: bool = False # Currently disabled for TPU training
    ema_decay: float = 0.99
    ema_update_interval: float = 0.02
    val_split: float = 0.05
    # warmup_steps: 15% or 50 steps

cfg = GPTConfig()

# Device selection: TPU > GPU > CPU
if HAS_XLA:
    device = torch_xla.device  # Use TPU via XLA
    cfg.dtype = torch.bfloat16  # BF16 optimal for TPU
    cfg.compile = False  # TPU: disable torch.compile (XLA is implicit, torch.compile adds overhead)
    cfg.dropout = 0.0  # TPU: disable dropout (interferes with XLA fusion, not needed for training stability)
    cfg.gradient_checkpointing = False  # XLA handles memory; checkpointing adds useless recomputation
    cfg.use_flash_attention = False  # TPU: SDPA is unfused, no FlashAttention support (~3-6x slower than GPU FA)
    cfg.sam = False  # SAM overhead not worth it on TPU
    print("Using TPU (XLA backend)")
elif torch.cuda.is_available():
    device = "cuda"
    cfg.dtype = torch.bfloat16  # BF16 for Ampere+
    print("Using CUDA GPU")
else:
    device = "cpu"
    cfg.dtype = torch.float32  # FP32 for CPU stability
    print("Using CPU")
    
# Backend optimizations (GPU only)
if torch.cuda.is_available() and not HAS_XLA:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True  # Auto-tune for faster convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matrix multiplications
    torch.backends.cudnn.allow_tf32 = True

# Initialize tokenizers at module level for imports
tokenizers = {}  # Dictionary to store multiple tokenizers by ID

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
        import numpy as np
        # Convert to numpy for faster indexing (avoids Python list overhead)
        self.token_stream = np.array(token_stream, dtype=np.int64)
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
        
        # Use torch.from_numpy (zero-copy) instead of torch.tensor (copies data)
        # Remove .copy() to enable true zero-copy operation
        seq = torch.from_numpy(self.token_stream[token_start:token_end])
        tgt = torch.from_numpy(self.token_stream[token_start + 1:token_end + 1])
        
        return seq, tgt


def create_data_loaders(token_stream, cfg, val_split=0.05, num_workers=None):
    """
    Create train and validation dataloaders with efficient batching.
    
    Args:
        token_stream: Complete token sequence
        cfg: Config object with batch_size and sequence_length
        val_split: Fraction of data for validation
        num_workers: Number of worker processes (auto-detected if None)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Auto-detect optimal num_workers
    if num_workers is None:
        num_workers = 4 if device == 'cuda' else 2
    
    total_tokens = len(token_stream)
    val_token_count = int(total_tokens * val_split)
    
    # Split data without materializing
    train_tokens = token_stream[val_token_count:]
    val_tokens = token_stream[:val_token_count]
    
    # Create datasets
    train_dataset = TokenSequenceDataset(train_tokens, cfg.sequence_length)
    val_dataset = TokenSequenceDataset(val_tokens, cfg.sequence_length)
    
    # Create dataloaders with optimizations
    # TPU prefers static shapes; use 0 workers if on TPU, else auto-detect
    num_workers_default = 0 if HAS_XLA else (4 if device == 'cuda' else 2)
    is_cuda = (device == 'cuda')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers_default,
        persistent_workers=(num_workers_default > 0),  # Avoid worker startup overhead
        pin_memory=is_cuda,  # Speed up GPU transfer (only for CUDA)
        drop_last=True,  # Drop incomplete last batch
        prefetch_factor=3 if num_workers_default > 0 else None,  # Increased prefetch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers_default,
        persistent_workers=(num_workers_default > 0),
        pin_memory=is_cuda,  # Speed up GPU transfer (only for CUDA)
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
    # Load dataset
    print("Loading dataset...")

    # Load exactly 150,000 examples from the train split
    ds = load_dataset("roneneldan/TinyStories", split="train[:150000]")

    # Extract the text field
    texts = [sample["text"] for sample in ds]

    # Join into a single string for tokenizer
    text = "\n".join(texts)

    print(f"Dataset loaded: {len(text)} characters")

    # Train tokenizer on Tiny Shakespeare
    print("\n" + "=" * 50)
    print("BPE Tokenizer...")
    print("=" * 50)
    
    # Create tokenizer with an ID (use min_frequency 8 for better compression)
    tokenizer_id = "tinystories"
    tokenizer = BPETokenizer(cfg.vocab_size, min_frequency=8, tokenizer_id=tokenizer_id)
    tokenizer_path = f"tokenizer_{tokenizer_id}.json"
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer '{tokenizer_id}' from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
    else:
        print(f"Training tokenizer '{tokenizer_id}'...")
        tokenizer.train(text, show_stats=True)
        tokenizer.save(tokenizer_path)
    
    # Store in dictionary
    tokenizers[tokenizer_id] = tokenizer
    
    print(f"Actual vocabulary size: {tokenizer.get_vocab_size()}")

    # Prepare training data - STREAMING BATCHES (no materialization)
    text_tokens = tokenizer.encode(text)
    
    total_tokens = len(text_tokens)
    compression_ratio = tokenizer.get_compression_ratio(text)
    
    print(f"Total tokens: {total_tokens} (compression ratio: {compression_ratio:.2f} chars/tokens)")

    # Create PyTorch DataLoaders for efficient batching and memory management
    train_loader, val_loader = create_data_loaders(
        text_tokens, 
        cfg, 
        val_split=cfg.val_split
        # num_workers auto-detected for GPU/CPU optimization
    )

    # Wrap dataloaders with MpDeviceLoader on TPU (eliminates Python iteration overhead)
    if HAS_XLA and MpDeviceLoader is not None:
        print("Wrapping dataloaders with MpDeviceLoader for XLA input pipeline fusion...")
        train_loader = MpDeviceLoader(train_loader, device)
        val_loader = MpDeviceLoader(val_loader, device)

    # Initialize model
    print("\nInitializing GPT model...")
    model = GPT(cfg).to(device=device, dtype=cfg.dtype)
    if cfg.compile:
        if HAS_XLA:
            print("TPU detected: Skipping torch.compile (XLA handles compilation)")
        else:
            print(f"Compiling model with mode='{cfg.compile_mode}' for maximum speed...")
            model = compile_model(model, mode=cfg.compile_mode)
    optimizer = build_optimizer(model, cfg)

    # EMA (Exponential Moving Average) for smooth weights
    ema_state_dict = None

    print("\n" + "=" * 50)
    print("Training Started...")
    print("=" * 50)
    model.train()

    warmp_steps = max(50, int(cfg.total_steps * 0.15))

    # Precompute full LR schedule as Python list (avoids device syncs on TPU)
    def get_lr_scale(step):
        if step < warmp_steps:
            return step / warmp_steps
        else:
            progress = (step - warmp_steps) / (cfg.total_steps - warmp_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    lr_schedule = [cfg.lr * get_lr_scale(i) for i in range(cfg.total_steps)]

    # BF16 training (no scaler needed)
    best_val_loss = float('inf')
    ema_state_dict = None
    ema_update_steps = max(1, int(cfg.total_steps * cfg.ema_update_interval))

    # Training loop: fixed number of steps (NO gradient accumulation on TPU)
    step = 0
    start_time = time.perf_counter()
    tokens_processed = 0
    loss_buffer = []  # Buffer losses to avoid device sync on TPU
    train_iter = iter(train_loader)  # Persistent iterator for sequential batches
    while step < cfg.total_steps:
        # Get batch from persistent iterator (NOT from for loop)
        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            # Reset iterator when dataset exhausted
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)
        
        # Move to device (MpDeviceLoader already places on device, but .to() is no-op then)
        # For non-TPU: this transfers GPU memory
        # For TPU+MpDeviceLoader: this is a no-op (already on device)
        if not HAS_XLA:
            x_batch = x_batch.to(device, dtype=torch.long)
            y_batch = y_batch.to(device, dtype=torch.long)
        else:
            # On TPU with MpDeviceLoader, tensors already on device
            # Just ensure correct dtype
            x_batch = x_batch.to(dtype=torch.long)
            y_batch = y_batch.to(dtype=torch.long)
        
        # Track tokens for throughput
        batch_tokens = x_batch.numel()
        
        # Set learning rate from precomputed schedule (Python list, no device sync)
        # TPU: update every 4 steps to reduce recompilation overhead (minimal accuracy loss)
        if step % 4 == 0:
            base_opt = optimizer.base if cfg.sam else optimizer
            base_opt.param_groups[0]['lr'] = lr_schedule[step]
        
        # Forward pass (NO autocast on TPU - XLA handles dtype natively)
        if HAS_XLA:
            # TPU: XLA handles mixed precision, no torch.autocast
            logits, _ = model(x_batch)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_batch.flatten(),
                label_smoothing=cfg.label_smoothing
            )
        else:
            # GPU: Use autocast for BF16
            with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                logits, _ = model(x_batch)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    y_batch.flatten(),
                    label_smoothing=cfg.label_smoothing
                )
        
        # Backward pass
        loss.backward()
        # Store loss WITHOUT calling .item() (avoids XLA device sync)
        # Detach to prevent XLA graph retention (loss is only for logging, not gradients)
        loss_buffer.append(loss.detach())
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # SAM: first step to perturb weights
        if cfg.sam:
            optimizer.first_step()
            
            # Recompute loss at perturbed weights (single batch for efficiency)
            try:
                x_batch_sam, y_batch_sam = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_batch_sam, y_batch_sam = next(train_iter)
            
            # Move to device (MpDeviceLoader already on device for TPU)
            if not HAS_XLA:
                x_batch_sam = x_batch_sam.to(device, dtype=torch.long)
                y_batch_sam = y_batch_sam.to(device, dtype=torch.long)
            else:
                x_batch_sam = x_batch_sam.to(dtype=torch.long)
                y_batch_sam = y_batch_sam.to(dtype=torch.long)
            
            if HAS_XLA:
                logits, _ = model(x_batch_sam)
                loss_perturbed = F.cross_entropy(
                    logits.flatten(0, 1),
                    y_batch_sam.flatten(),
                    label_smoothing=cfg.label_smoothing
                )
            else:
                with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                    logits, _ = model(x_batch_sam)
                    loss_perturbed = F.cross_entropy(
                        logits.flatten(0, 1),
                        y_batch_sam.flatten(),
                        label_smoothing=cfg.label_smoothing
                    )
            loss_perturbed.backward()
            optimizer.second_step()
        else:
            base_opt.step()
            base_opt.zero_grad()
        
        # Mark XLA step (critical for TPU): tells XLA to finalize this training step
        if HAS_XLA:
            xm.mark_step()
        
        tokens_processed += batch_tokens
        
        # Update EMA (infrequently, every 1-3% of total steps)
        if cfg.use_ema and step % ema_update_steps == 0:
            if ema_state_dict is None:
                ema_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    ema_state_dict[k] = cfg.ema_decay * ema_state_dict[k] + (1 - cfg.ema_decay) * v
        
        # Periodic validation
        if (step + 1) % cfg.val_interval == 0:
            model.eval()
            
            # Use EMA weights for validation if available
            if cfg.use_ema and ema_state_dict is not None:
                original_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state_dict)
            
            # Batch validation losses (avoid .item() calls until final reduction)
            # TPU: validate on only 2 batches (full validation is too expensive with many mark_step calls)
            # GPU: validate on full dataset
            val_losses = []
            max_val_batches = 2 if HAS_XLA else float('inf')
            val_batch_count = 0
            with torch.no_grad():
                # Use DataLoader for validation (more efficient)
                for x_vbatch, y_vbatch in val_loader:
                    if val_batch_count >= max_val_batches:
                        break
                    
                    # MpDeviceLoader already on device; only dtype conversion needed on TPU
                    if not HAS_XLA:
                        x_vbatch = x_vbatch.to(device, dtype=torch.long)
                        y_vbatch = y_vbatch.to(device, dtype=torch.long)
                    else:
                        x_vbatch = x_vbatch.to(dtype=torch.long)
                        y_vbatch = y_vbatch.to(dtype=torch.long)
                    logits, _ = model(x_vbatch)
                    val_loss_batch = F.cross_entropy(
                        logits.flatten(0, 1),
                        y_vbatch.flatten()
                    )
                    # Store tensor, don't sync with .item()
                    val_losses.append(val_loss_batch)
                    val_batch_count += 1
            
            # Mark XLA step after validation (batch all validation graphs together)
            if HAS_XLA:
                xm.mark_step()
            
            # Single reduction at end (one device sync instead of per-batch syncs)
            if val_losses:
                val_loss = torch.stack(val_losses).mean().item()
            else:
                val_loss = float('inf')
            
            # Restore original weights before saving
            if cfg.use_ema and ema_state_dict is not None:
                model.load_state_dict(original_state)
            
            # Save best model (XLA-aware checkpoint if on TPU)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if HAS_XLA:
                    xm.save(model.state_dict(), "model/best_model.pt")
                    if cfg.use_ema:
                        xm.save(ema_state_dict, "model/best_ema_model.pt")
                else:
                    torch.save(model.state_dict(), "model/best_model.pt")
                    if cfg.use_ema:
                        torch.save(ema_state_dict, "model/best_ema_model.pt")
            
            model.train()
            elapsed = time.perf_counter() - start_time
            # Average the buffered losses (single device sync)
            train_loss = torch.stack(loss_buffer).mean().item() if loss_buffer else 0.0
            loss_buffer.clear()
            throughput = tokens_processed / elapsed if elapsed > 0 else 0
            print(f"Step {step+1:5d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Throughput {throughput:.0f} tok/s")
        
        step += 1