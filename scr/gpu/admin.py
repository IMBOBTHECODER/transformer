# admin.py (GPU)

import os
import time
import math
import torch
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from core import GPT, BPETokenizer
from utils import build_optimizer
import gc
from datasets import load_dataset


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
    yarn_scale: float = 1.0

    # GPU settings
    dtype: torch.dtype = torch.float32
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    use_rope: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # Activation and optimization
    activation: str = "swiglu"
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 128
    sequence_length: int = 512

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    # SAM optimizer
    sam: bool = False
    sam_rho: float = 0.08

    # Training schedule
    init_std: float = 0.02
    id: str = "tinystories"
    total_steps: int = 5_000
    val_interval: int = 200
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_update_interval: float = 0.02
    val_split: float = 0.05


cfg = GPTConfig()


# =========================================================
# DEVICE SETUP
# =========================================================
def setup_device():
    """Configure device and optimize settings."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0).lower()

        # Modern GPUs (A100, H100, RTX3090+): use BF16 and optimizations
        if "a100" in gpu_name or "h100" in gpu_name or "rtx" in gpu_name:
            cfg.dtype = torch.bfloat16
            cfg.use_flash_attention = True
            print(f"✓ GPU: {gpu_name.upper()} | BF16 | Flash Attention")
        else:
            # Older GPUs (T4, K80, P100): limited to FP32
            cfg.dtype = torch.float32
            cfg.compile = False
            cfg.gradient_checkpointing = True
            cfg.batch_size = min(cfg.batch_size, 64)
            print(f"✓ GPU: {gpu_name.upper()} | FP32 | batch={cfg.batch_size}")

        # GPU matrix operation optimizations
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"
        cfg.dtype = torch.float32
        cfg.compile = False
        print("✓ CPU mode")

    return device


device = setup_device()
print(f"Device: {device}\n")


# =========================================================
# TOKENIZER & DATASET PREPARATION
# =========================================================
def prepare_tokenizer_and_dataset(vocab_size, tokenizer_id, num_samples=250000):
    """Train tokenizer and tokenize dataset."""
    print("=" * 50)
    print("Tokenizer & Dataset Preparation")
    print("=" * 50)

    tokenizer_path = f"tokenizer/tokenizer_{tokenizer_id}.json"
    tokens_path = f"tokenizer/tokens_{tokenizer_id}.npy"

    # Skip if both already exist
    if os.path.exists(tokenizer_path) and os.path.exists(tokens_path):
        print(f"✓ Tokenizer and tokens already exist")
        tokenizer = BPETokenizer(vocab_size, tokenizer_id=tokenizer_id)
        tokenizer.load(tokenizer_path)
        text_tokens = np.load(tokens_path)
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print(f"  Total tokens: {len(text_tokens):,}\n")
        return tokenizer, text_tokens

    # Load dataset
    print(f"Loading TinyStories dataset ({num_samples:,} samples)...")
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{num_samples}]")
    print(f"✓ Dataset loaded: {len(ds):,} samples")

    # Stream text in chunks
    def stream_text(dataset, chunk_size=16384):
        buffer = []
        for sample in dataset:
            buffer.append(sample["text"])
            if len(buffer) >= chunk_size:
                yield "\n".join(buffer)
                buffer.clear()
        if buffer:
            yield "\n".join(buffer)

    print("Streaming text chunks...")
    chunks = list(stream_text(ds))
    text = "\n".join(chunks)
    del chunks
    gc.collect()

    # Train tokenizer
    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer = BPETokenizer(vocab_size, min_frequency=8, tokenizer_id=tokenizer_id)
    tokenizer.train(text, show_stats=True)

    # Save tokenizer
    print(f"Saving tokenizer to {tokenizer_path}...")
    tokenizer.save(tokenizer_path)
    print(f"✓ Tokenizer saved\n")

    # Tokenize dataset
    print("Tokenizing dataset...")
    chunks = []
    for i, sample in enumerate(ds):
        if i % 50000 == 0:
            print(f"  Tokenized {i:,} samples...")
        tokens = tokenizer.encode(sample["text"])
        chunks.append(np.asarray(tokens, dtype=np.uint16))

    text_tokens = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    del chunks, text, ds
    gc.collect()

    print(f"Total tokens: {len(text_tokens):,}")

    # Save tokens
    print(f"Saving tokenized dataset to {tokens_path}...")
    np.save(tokens_path, text_tokens)
    print(f"✓ Dataset tokenized and saved\n")

    return tokenizer, text_tokens


# =========================================================
# DATASET & DATALOADERS
# =========================================================
class TokenSequenceDataset(Dataset):
    """Efficient token sequence dataset with zero-copy numpy indexing."""

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

        if end >= len(self.token_stream):
            end = len(self.token_stream) - 1
            start = max(0, end - self.seq_len)

        seq = torch.from_numpy(self.token_stream[start:end].copy())
        tgt = torch.from_numpy(self.token_stream[start + 1:end + 1].copy())
        return seq, tgt


def create_data_loaders(token_stream, cfg, val_split=0.05):
    """Create dataloaders with GPU optimizations."""
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
        shuffle=True,
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

    print(f"Train: {len(train_dataset):,} sequences | Val: {len(val_dataset):,} sequences")
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
        if use_cache and cache is not None:
            logits, cache = model(idx[:, -1:], cache)
        else:
            logits, cache = model(idx, cache)

        logits = logits[:, -1:] / temperature

        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits_mask = torch.full_like(logits, float('-inf'))
            logits_mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_mask

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(1), 1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


# =========================================================
# CONTROL PANEL
# =========================================================
class Control:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.ema_state_dict = None

    def display(self):
        print("\x1b[36m .d8888b.                    888                    888      8888888b.                            888")
        print("d88P  Y88b                   888                    888      888   Y88b                           888 ")
        print("888    888                   888                    888      888    888                           888 ")
        print("888         .d88b.  88888b.  888888 888d888 .d88b.  888      888   d88P 8888b.  88888b.   .d88b.  888 ")
        print("888        d88\"\"88b 888 \"88b 888    888P\"  d88\"\"88b 888      8888888P\"     \"88b 888 \"88b d8P  Y8b 888 ")
        print("888    888 888  888 888  888 888    888    888  888 888      888       .d888888 888  888 88888888 888 ")
        print("Y88b  d88P Y88..88P 888  888 Y88b.  888    Y88..88P 888      888       888  888 888  888 Y8b.     888 ")
        print(" \"Y8888P\"   \"Y88P\"  888  888  \"Y888 888     \"Y88P\"  888      888       \"Y888888 888  888  \"Y8888  888 \x1b[0m")

        print("━" * 50)
        print(f"  Device: \x1b[94m{device}\x1b[0m | Dtype: \x1b[94m{cfg.dtype}\x1b[0m")
        print(f"  Model ID: \x1b[94m{cfg.id}\x1b[0m | Vocab: \x1b[94m{cfg.vocab_size:,}\x1b[0m")
        print(f"  Batch: \x1b[94m{cfg.batch_size}\x1b[0m | Seq Length: \x1b[94m{cfg.sequence_length}\x1b[0m")
        print(f"  Steps: \x1b[94m{cfg.total_steps:,}\x1b[0m | LR: \x1b[94m{cfg.lr}\x1b[0m")
        print(f"  EMA: \x1b[94m{cfg.use_ema}\x1b[0m | SAM: \x1b[94m{cfg.sam}\x1b[0m")
        print("━" * 50 + "\n")

    def tokenize(self):
        print("GPU TOKENIZATION")
        print("━" * 50 + "\n")
        tokenizer, text_tokens = prepare_tokenizer_and_dataset(cfg.vocab_size, cfg.id)
        print(f"Ready for training: {len(text_tokens):,} tokens\n")

        self.train_loader, self.val_loader = create_data_loaders(
            text_tokens, cfg, val_split=cfg.val_split
        )

    def load_checkpoint(self, checkpoint_path=None):
        """Load a previously saved model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = f"model/best_model_{cfg.id}.pt"

        if not os.path.exists(checkpoint_path):
            print(f"⚠ Checkpoint not found at {checkpoint_path}")
            return None

        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Remove torch.compile wrapper prefix if present
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        print(f"✓ Checkpoint loaded successfully\n")
        return state_dict

    def train(self, resume=False, checkpoint_path=None):
        print("GPU TRAINING")
        print("━" * 50 + "\n")

        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("DataLoaders not initialized. Call tokenize() first.")

        # Initialize model
        print("Initializing GPT model...")
        self.model = GPT(cfg).to(device=device, dtype=cfg.dtype)

        # Load checkpoint if resuming
        if resume:
            if checkpoint_path is None:
                checkpoint_path = f"model/best_model_{cfg.id}.pt"
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
                print("Resuming training from checkpoint...")
            else:
                print(f"⚠ Checkpoint not found. Starting fresh...")

        # Compile model for speedup
        if cfg.compile:
            from core import compile_model
            self.model = compile_model(self.model, cfg.compile_mode)
            print("✓ Model compiled with torch.compile")

        self.optimizer = build_optimizer(self.model, cfg)
        self.model.train()

        # Learning rate schedule
        warmup_steps = max(50, int(cfg.total_steps * 0.15))

        def get_lr(step):
            if step < warmup_steps:
                return cfg.lr * step / warmup_steps
            progress = (step - warmup_steps) / (cfg.total_steps - warmup_steps)
            return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        lr_schedule = [get_lr(i) for i in range(cfg.total_steps)]

        best_val_loss = float('inf')
        ema_update_steps = max(1, int(cfg.total_steps * cfg.ema_update_interval))

        # =========================================================
        # TRAINING LOOP
        # =========================================================
        step = 0
        start_time = time.perf_counter()
        tokens_processed = 0
        loss_buffer = []
        train_iter = iter(self.train_loader)

        print(f"Starting training for {cfg.total_steps:,} steps...")
        print(f"Warmup: {warmup_steps} steps | Val interval: {cfg.val_interval}\n")

        while step < cfg.total_steps:
            try:
                x_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x_batch, y_batch = next(train_iter)

            x_batch = x_batch.to(device=device, dtype=torch.long, non_blocking=True)
            y_batch = y_batch.to(device=device, dtype=torch.long, non_blocking=True)

            batch_tokens = x_batch.numel()

            # Update learning rate
            base_opt = self.optimizer.base if cfg.sam else self.optimizer
            for param_group in base_opt.param_groups:
                param_group['lr'] = lr_schedule[step]

            # Forward pass
            logits, _ = self.model(x_batch)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_batch.flatten(),
                label_smoothing=cfg.label_smoothing
            )

            # Backward pass
            loss.backward()
            loss_buffer.append(loss.detach())

            # Gradient clipping
            if cfg.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

            # Optimizer step
            if cfg.sam:
                self.optimizer.first_step()
                logits, _ = self.model(x_batch)
                loss_perturbed = F.cross_entropy(
                    logits.flatten(0, 1),
                    y_batch.flatten(),
                    label_smoothing=cfg.label_smoothing
                )
                loss_perturbed.backward()
                self.optimizer.second_step()
                self.optimizer.zero_grad()
            else:
                base_opt.step()
                base_opt.zero_grad(set_to_none=True)

            tokens_processed += batch_tokens

            # Update EMA
            if cfg.use_ema and step % ema_update_steps == 0:
                if self.ema_state_dict is None:
                    self.ema_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    for k, v in self.model.state_dict().items():
                        self.ema_state_dict[k].mul_(cfg.ema_decay).add_(v, alpha=1 - cfg.ema_decay)

            # =========================================================
            # VALIDATION & CHECKPOINTING
            # =========================================================
            if (step + 1) % cfg.val_interval == 0:
                if cfg.use_ema and self.ema_state_dict is not None:
                    original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.model.load_state_dict(self.ema_state_dict)

                val_losses = []
                max_val_batches = 5
                val_batch_count = 0
                
                with torch.no_grad():
                    for x_vbatch, y_vbatch in self.val_loader:
                        if val_batch_count >= max_val_batches:
                            break

                        x_vbatch = x_vbatch.to(device=device, dtype=torch.long, non_blocking=True)
                        y_vbatch = y_vbatch.to(device=device, dtype=torch.long, non_blocking=True)
                        logits, _ = self.model(x_vbatch)
                        val_loss_batch = F.cross_entropy(logits.flatten(0, 1), y_vbatch.flatten())
                        val_losses.append(val_loss_batch)
                        val_batch_count += 1

                val_loss = torch.stack(val_losses).mean().item() if val_losses else float('inf')

                if cfg.use_ema and self.ema_state_dict is not None:
                    self.model.load_state_dict(original_state)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs("model", exist_ok=True)
                    torch.save(self.model.state_dict(), f"model/best_model_{cfg.id}.pt")
                    if cfg.use_ema and self.ema_state_dict is not None:
                        torch.save(self.ema_state_dict, f"model/best_ema_model_{cfg.id}.pt")

                elapsed = time.perf_counter() - start_time
                train_loss = torch.stack(loss_buffer).mean().item() if loss_buffer else 0.0
                loss_buffer.clear()
                throughput = tokens_processed / elapsed if elapsed > 0 else 0
                lr_current = lr_schedule[step]
                print(f"Step {step+1:5d} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr_current:.2e} | {throughput:,.0f} tok/s")

            step += 1

        print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    control = Control()
    control.display()
    control.tokenize()

    # Set resume=True to continue training from checkpoint
    # control.train(resume=True, checkpoint_path="model/best_model_tinystories.pt")
    control.train(resume=False)
