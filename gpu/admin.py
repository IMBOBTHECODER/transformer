# admin.py (GPU-optimized training)
"""
GPU-optimized transformer training with CUDA optimizations and flash attention.
Auto-detects GPU capabilities and adapts configuration for optimal performance.
"""

import os
import sys
import gc
import time
import torch
import math
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Suppress tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from core import compile_model, GPT, BPETokenizer
from utils import build_optimizer

# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 18_000
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 8
    mlp_ratio: int = 4
    yarn_scale: float = 1.0  # Position scaling for extended context during generation

    # Performance settings
    dtype: torch.dtype = torch.float32  # float32 | bfloat16 | float16
    compile: bool = True
    compile_mode: str = "reduce-overhead"  # Training: reduce-overhead | Inference: max-autotune
    use_rope: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # Activation function
    activation: str = "swiglu"  # relu | relu2 | gelu | silu | swiglu | geglu | reglu | mish | snake

    # Optimization hyperparameters
    optimizer: str = "adamw"  # adamw | lion | sgd
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 128
    sequence_length: int = 512

    # Training regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1

    # Sharpness-Aware Minimization (SAM)
    sam: bool = False
    sam_rho: float = 0.08

    # Initialization and tracking
    init_std: float = 0.02
    id: str = "tinystories"

    # Training schedule
    total_steps: int = 5_000
    val_interval: int = 200
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_update_interval: float = 0.02
    val_split: float = 0.05
    # Note: warmup is 15% of total_steps or 50 steps (whichever is larger)

cfg = GPTConfig()

# =====================================================
# GPU DETECTION & CONFIGURATION
# =====================================================
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0).lower()

    # Modern GPUs (A100, H100, RTX3090+): use BF16 and optimizations
    if "a100" in gpu_name or "h100" in gpu_name or "rtx" in gpu_name:
        cfg.dtype = torch.bfloat16
        cfg.use_flash_attention = True
        print(f"✓ GPU: {gpu_name.upper()} | BF16 | Flash Attention | torch.compile")

    # Older GPUs (T4, K80, P100): limited to FP32
    else:
        cfg.dtype = torch.float32
        cfg.compile = False
        cfg.gradient_checkpointing = True
        cfg.batch_size = min(cfg.batch_size, 64)
        print(f"✓ GPU: {gpu_name.upper()} | FP32 | batch={cfg.batch_size} | grad_checkpt")
else:
    device = "cpu"
    cfg.dtype = torch.float32
    cfg.compile = False
    print("✓ CPU mode")

# Enable GPU matrix operation optimizations
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # TF32 for FP32 matrix ops
    torch.backends.cudnn.benchmark = True  # Auto-tune CUDNN kernels
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 acceleration
    torch.backends.cudnn.allow_tf32 = True

tokenizers = {}

# =========================================================
# DATASET AND DATALOADER
# =========================================================

class TokenSequenceDataset(Dataset):
    """Efficient token sequence dataset with minimal memory overhead."""

    def __init__(self, token_stream, seq_len, stride=None):
        self.token_stream = np.array(token_stream, dtype=np.int64)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2
        self.num_sequences = max(1, (len(token_stream) - seq_len) // self.stride)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len

        # Clamp to array bounds
        if end >= len(self.token_stream):
            end = len(self.token_stream) - 1
            start = max(0, end - self.seq_len)

        # Zero-copy numpy-to-tensor conversion
        seq = torch.from_numpy(self.token_stream[start:end])
        tgt = torch.from_numpy(self.token_stream[start + 1:end + 1])
        return seq, tgt


def create_data_loaders(token_stream, cfg, val_split=0.05):
    """Create dataloaders with GPU optimizations (pinned memory, persistent workers)."""
    total_tokens = len(token_stream)
    val_token_count = int(total_tokens * val_split)

    train_tokens = token_stream[val_token_count:]
    val_tokens = token_stream[:val_token_count]

    train_dataset = TokenSequenceDataset(train_tokens, cfg.sequence_length)
    val_dataset = TokenSequenceDataset(val_tokens, cfg.sequence_length)

    # GPU optimizations: pinned memory and persistent workers
    num_workers = 4 if device == 'cuda' else 2
    is_cuda = (device == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=is_cuda,
        drop_last=True,
        prefetch_factor=3 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=is_cuda,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset)} sequences | Val: {len(val_dataset)} sequences")
    print(f"DataLoaders: batch={cfg.batch_size}, workers={num_workers}, pinned={is_cuda}\n")
    return train_loader, val_loader


# =========================================================
# TEXT GENERATION
# =========================================================

@torch.no_grad()
def generate(model, idx, steps, temperature=0.9, top_k=40, use_cache=True):
    """Generate text with temperature and top-k sampling using KV cache."""
    model.eval()
    cache = None

    for _ in range(steps):
        # Use cached KV if available to avoid recomputation
        if use_cache and cache is not None:
            logits, cache = model(idx[:, -1:], cache)  # Only process last token
        else:
            logits, cache = model(idx, cache)

        logits = logits[:, -1:] / temperature

        # Top-k filtering: keep only top-k most likely tokens
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
    print("\n" + "=" * 60)
    print("TRAINING TRANSFORMER ON GPU")
    print("=" * 60)

    # =====================================================
    # LOAD DATASET
    # =====================================================
    print("\nLoading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:250000]")
    print(f"Dataset: {len(ds)} samples\n")

    # =====================================================
    # TOKENIZER
    # =====================================================
    tokenizer_id = cfg.id
    tokenizer = BPETokenizer(cfg.vocab_size, min_frequency=8, tokenizer_id=tokenizer_id)
    tokenizer_path = f"tokenizer_{tokenizer_id}.json"

    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        # Stream text in chunks to avoid memory spike
        def stream_text(dataset, chunk_size=16384):
            buffer = []
            for sample in dataset:
                buffer.append(sample["text"])
                if len(buffer) >= chunk_size:
                    yield "\n".join(buffer)
                    buffer.clear()
            if buffer:
                yield "\n".join(buffer)

        print("Training BPE tokenizer...")
        text = "\n".join(stream_text(ds))
        tokenizer.train(text, show_stats=True)
        tokenizer.save(tokenizer_path)
        del text

    tokenizers[tokenizer_id] = tokenizer
    print(f"✓ Vocab size: {tokenizer.get_vocab_size()}\n")

    # =====================================================
    # TOKENIZATION
    # =====================================================
    print("Tokenizing dataset (streaming)...")

    def stream_tokenize(dataset, tokenizer, chunk_size=16384):
        buffer = []
        for sample in dataset:
            buffer.append(sample["text"])
            if len(buffer) >= chunk_size:
                yield tokenizer.encode("\n".join(buffer))
                buffer.clear()
        if buffer:
            yield tokenizer.encode("\n".join(buffer))

    chunks = []
    for chunk_tokens in stream_tokenize(ds, tokenizer):
        chunks.append(np.asarray(chunk_tokens, dtype=np.uint16))

    text_tokens = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    del chunks, ds
    gc.collect()

    print(f"Total tokens: {len(text_tokens):,}\n")

    # =====================================================
    # CREATE DATALOADERS
    # =====================================================
    train_loader, val_loader = create_data_loaders(text_tokens, cfg, val_split=cfg.val_split)

    # =====================================================
    # INITIALIZE MODEL
    # =====================================================
    model = GPT(cfg).to(device=device, dtype=cfg.dtype)
    if cfg.compile:
        print(f"Compiling model ({cfg.compile_mode})...")
        model = compile_model(model, mode=cfg.compile_mode)
    optimizer = build_optimizer(model, cfg)

    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    model.train()

    # Warmup: first 15% of steps or 50 steps (whichever is larger)
    warmup_steps = max(50, int(cfg.total_steps * 0.15))

    def get_lr_scale(step):
        """Cosine annealing with linear warmup."""
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (cfg.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_schedule = [cfg.lr * get_lr_scale(i) for i in range(cfg.total_steps)]

    # Training state
    best_val_loss = float('inf')
    ema_state_dict = None
    ema_update_steps = max(1, int(cfg.total_steps * cfg.ema_update_interval))

    step = 0
    start_time = time.perf_counter()
    tokens_processed = 0
    loss_buffer = []
    train_iter = iter(train_loader)

    # GPU mixed precision training: uses FP16/BF16 for compute, FP32 for model weights
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    while step < cfg.total_steps:
        # Load next batch, cycle through train data if needed
        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)

        # Non-blocking GPU transfer speeds up data loading (CPU doesn't wait for GPU)
        x_batch = x_batch.to(device, dtype=torch.long, non_blocking=True)
        y_batch = y_batch.to(device, dtype=torch.long, non_blocking=True)
        batch_tokens = x_batch.numel()

        # Update learning rate based on schedule
        base_opt = optimizer.base if cfg.sam else optimizer
        lr = lr_schedule[step]
        base_opt.param_groups[0]['lr'] = lr

        # Forward pass with autocast: compute in lower precision, keep weights in FP32
        with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
            logits, _ = model(x_batch)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_batch.flatten(),
                label_smoothing=cfg.label_smoothing
            )

        # Backward pass: GPU uses GradScaler for mixed precision
        if device == 'cuda' and scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if cfg.sam:
                # SAM: perturb weights, compute loss on perturbed weights, then step
                optimizer.first_step()
                with torch.autocast(device_type=device, dtype=cfg.dtype):
                    logits, _ = model(x_batch)
                    loss_perturbed = F.cross_entropy(
                        logits.flatten(0, 1),
                        y_batch.flatten(),
                        label_smoothing=cfg.label_smoothing
                    )
                scaler.scale(loss_perturbed).backward()
                scaler.unscale_(base_opt)
                optimizer.second_step()
                optimizer.zero_grad()
            else:
                # Standard AdamW/Lion: unscale gradients, then step
                scaler.unscale_(base_opt)
                scaler.step(base_opt)
                scaler.update()
                base_opt.zero_grad()
        else:
            # CPU or no scaler: standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if cfg.sam:
                optimizer.first_step()
                with torch.autocast(device_type=device, dtype=cfg.dtype if device == 'cuda' else torch.float32):
                    logits, _ = model(x_batch)
                    loss_perturbed = F.cross_entropy(
                        logits.flatten(0, 1),
                        y_batch.flatten(),
                        label_smoothing=cfg.label_smoothing
                    )
                loss_perturbed.backward()
                optimizer.second_step()
                optimizer.zero_grad()
            else:
                base_opt.step()
                base_opt.zero_grad()

        loss_buffer.append(loss.detach())
        tokens_processed += batch_tokens
        
        # Periodically clear GPU cache to prevent memory fragmentation (every 100 steps)
        if device == 'cuda' and (step + 1) % 100 == 0:
            torch.cuda.empty_cache()
        
        # =====================================================
        # EMA WEIGHTS (exponential moving average)
        # =====================================================
        if cfg.use_ema and step % ema_update_steps == 0:
            if ema_state_dict is None:
                # Clone weights for EMA initialization (detach to break computational graph)
                ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                # Use in-place ops to reduce memory allocations during EMA update
                for k, v in model.state_dict().items():
                    ema_state_dict[k].mul_(cfg.ema_decay).add_(v, alpha=1 - cfg.ema_decay)
        
        # =====================================================
        # VALIDATION & CHECKPOINTING
        # =====================================================
        if (step + 1) % cfg.val_interval == 0:
            model.eval()

            # If using EMA, test with EMA weights
            if cfg.use_ema and ema_state_dict is not None:
                original_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state_dict)

            # Evaluate on validation set
            val_losses = []
            with torch.no_grad():
                for x_vbatch, y_vbatch in val_loader:
                    x_vbatch = x_vbatch.to(device, dtype=torch.long)
                    y_vbatch = y_vbatch.to(device, dtype=torch.long)
                    logits, _ = model(x_vbatch)
                    val_loss = F.cross_entropy(logits.flatten(0, 1), y_vbatch.flatten())
                    val_losses.append(val_loss)

            val_loss = torch.stack(val_losses).mean().item() if val_losses else float('inf')

            # Restore original weights after EMA eval
            if cfg.use_ema and ema_state_dict is not None:
                model.load_state_dict(original_state)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("model", exist_ok=True)
                torch.save(model.state_dict(), f"model/best_model_{cfg.id}.pt")
                if cfg.use_ema and ema_state_dict is not None:
                    torch.save(ema_state_dict, f"model/best_ema_model_{cfg.id}.pt")

            model.train()
            elapsed = time.perf_counter() - start_time
            train_loss = torch.stack(loss_buffer).mean().item() if loss_buffer else 0.0
            loss_buffer.clear()
            throughput = tokens_processed / elapsed if elapsed > 0 else 0
            print(f"Step {step+1:5d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | {throughput:.0f} tok/s")

        step += 1