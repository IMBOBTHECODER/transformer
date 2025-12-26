# admin.py (TPU)

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

# TPU imports
import torch_xla.core.xla_model as xm
from torch_xla.distributed.parallel_loader import MpDeviceLoader

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

    # TPU settings: BF16, no compile, no dropout, full MHA
    dtype: torch.dtype = torch.bfloat16
    compile: bool = False  # XLA handles compilation
    compile_mode: str = "reduce-overhead"
    use_rope: bool = True
    gradient_checkpointing: bool = False  # XLA handles memory
    use_flash_attention: bool = False  # SDPA unfused on TPU

    # Activation and optimization
    activation: str = "swiglu"
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 128
    sequence_length: int = 512

    # Regularization
    dropout: float = 0.0  # TPU: disabled (interferes with XLA fusion)
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False  # TPU: disabled (unnecessary with BF16)

    # SAM optimizer
    sam: bool = False
    sam_rho: float = 0.08

    # Training schedule
    init_std: float = 0.02
    id: str = "tinystories"
    total_steps: int = 5_000
    val_interval: int = 200
    use_ema: bool = False
    ema_decay: float = 0.99
    ema_update_interval: float = 0.02
    val_split: float = 0.05


cfg = GPTConfig()
# TPU device setup
device = xm.xla_device()
print(f"TPU device: {device}")


# =========================================================
# TOKENIZER & DATASET PREPARATION
# =========================================================
def prepare_tokenizer_and_dataset(vocab_size, tokenizer_id, amount=None):
    """Train tokenizer and tokenize dataset in one go."""
    print("=" * 50)
    print("Tokenizer & Dataset Preparation")
    print("=" * 50)
    
    tokenizer_path = f"tokenizer_{tokenizer_id}.json"
    tokens_path = f"tokens_{tokenizer_id}.npy"
    
    # Skip if both already exist
    if os.path.exists(tokenizer_path) and os.path.exists(tokens_path):
        print(f"✓ Tokenizer and tokens already exist")
        tokenizer = BPETokenizer(vocab_size, tokenizer_id=tokenizer_id)
        tokenizer.load(tokenizer_path)
        text_tokens = np.load(tokens_path)
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print(f"  Total tokens: {len(text_tokens)}\n")
        return tokenizer, text_tokens
    
    # Load dataset
    print("Loading TinyStories dataset (250k samples)...")
    ds = load_dataset("roneneldan/TinyStories", split="train[:250000]")
    print(f"✓ Dataset loaded: {len(ds)} samples")
    
    # Stream text in chunks
    def stream_text(dataset, chunk_size=16384):
        """Stream text in chunks to avoid memory overload."""
        buffer = []
        for sample in dataset:
            buffer.append(sample["text"])
            if len(buffer) >= chunk_size:
                yield "\n".join(buffer)
                buffer.clear()
        if buffer:
            yield "\n".join(buffer)
    
    # Stream and concatenate all text
    print("Streaming text chunks...")
    chunks = list(stream_text(ds))
    text = "\n".join(chunks)
    del chunks
    gc.collect()
    
    # Train tokenizer
    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer = BPETokenizer(vocab_size, min_frequency=8, tokenizer_id=tokenizer_id)
    tokenizer.train(text, show_stats=True)
    
    # Save tokenizer
    print(f"Saving tokenizer to {tokenizer_path}...")
    tokenizer.save(tokenizer_path)
    print(f"✓ Tokenizer saved\n")
    
    # Tokenize dataset
    print("Tokenizing dataset (streaming)...")
    chunks = []
    for i, sample in enumerate(ds):
        if i % 50000 == 0:
            print(f"  Tokenized {i} samples...")
        tokens = tokenizer.encode(sample["text"])
        chunks.append(np.asarray(tokens, dtype=np.uint16))
    
    # Concatenate all tokens
    text_tokens = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
    del chunks, text, ds
    gc.collect()
    
    total_tokens = len(text_tokens)
    print(f"Total tokens: {total_tokens}")
    
    # Save tokens
    print(f"Saving tokenized dataset to {tokens_path}...")
    np.save(tokens_path, text_tokens)
    print(f"✓ Dataset tokenized and saved\n")
    
    return tokenizer, text_tokens


# =========================================================
# TPU METRICS TRACKING
# =========================================================
class TPUMetrics:
    """Track training metrics on TPU."""
    
    def __init__(self):
        self.step_count = 0
        self.loss_sum = 0.0
        
    def log(self, loss: float) -> None:
        """Record loss value."""
        self.step_count += 1
        self.loss_sum += loss
        
    def get_avg_loss(self) -> float:
        """Get average loss."""
        if self.step_count == 0:
            return 0.0
        return self.loss_sum / self.step_count
        
    def reset(self) -> None:
        """Reset metrics for next batch."""
        self.step_count = 0
        self.loss_sum = 0.0


# =========================================================
# XLA GRAPH COMPILATION WARMUP
# =========================================================
def warmup_xla_graph(model, batch_size, seq_len, device):
    """Warm up XLA compilation once at startup for consistent training speed."""
    print("Warming up XLA graph compilation...")
    # Use zeros to guarantee token distribution matches real data
    x_sample = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i in range(3):
            _ = model(x_sample)
            xm.mark_step()
            print(f"  Warmup step {i+1}/3")
    print("Warmup complete\n")


# =========================================================
# DATASET & DATALOADERS
# =========================================================
class TokenSequenceDataset(Dataset):
    """Efficient token sequence dataset with zero-copy numpy indexing."""
    
    def __init__(self, token_stream, seq_len, stride=None):
        self.token_stream = np.array(token_stream, dtype=np.int64)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.num_sequences = max(1, (len(token_stream) - seq_len) // self.stride)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        
        if end >= len(self.token_stream):
            end = len(self.token_stream) - 1
            start = max(0, end - self.seq_len)
        
        # Zero-copy numpy-to-tensor conversion
        seq = torch.from_numpy(self.token_stream[start:end])
        tgt = torch.from_numpy(self.token_stream[start + 1:end + 1])
        return seq, tgt


def create_data_loaders(token_stream, cfg, val_split=0.05):
    """Create dataloaders with MpDeviceLoader for XLA input pipeline fusion."""
    total_tokens = len(token_stream)
    val_token_count = int(total_tokens * val_split)
    
    train_tokens = token_stream[val_token_count:]
    val_tokens = token_stream[:val_token_count]
    
    train_dataset = TokenSequenceDataset(train_tokens, cfg.sequence_length)
    val_dataset = TokenSequenceDataset(val_tokens, cfg.sequence_length)
    
    # Base dataloaders (no workers on TPU - MpDeviceLoader handles parallelism)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    
    # Wrap with MpDeviceLoader for input pipeline fusion on TPU
    train_loader, val_loader = MpDeviceLoader(train_loader, device), MpDeviceLoader(val_loader, device)
    
    print(f"Train: {len(train_dataset)} sequences | Val: {len(val_dataset)} sequences")
    print(f"DataLoaders: batch={cfg.batch_size} | MpDeviceLoader (XLA fusion)\n")
    return train_loader, val_loader


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

        print("━" * 30)
        print(f"    Device: \x1b[94m{device}\x1b[0m")
        print(f"    Model ID: \x1b[94m{cfg.id}\x1b[0m")
        print(f"    Vocabulary Size: \x1b[94m{cfg.vocab_size}\x1b[0m")
        print(f"    Batch Size: \x1b[94m{cfg.batch_size}\x1b[0m")
        print(f"    Sequence Length: \x1b[94m{cfg.sequence_length}\x1b[0m")
        print(f"    Total Steps: \x1b[94m{cfg.total_steps}\x1b[0m")
        print(f"    Use EMA: \x1b[94m{cfg.use_ema}\x1b[0m | Decay: \x1b[94m{cfg.ema_decay}\x1b[0m")
        print(f"    SAM Optimizer: \x1b[94m{cfg.sam}\x1b[0m | Rho: \x1b[94m{cfg.sam_rho}\x1b[0m")
        print("━" * 30 + "\n")

    def tokenize(self):
        print("TPU TOKENIZATION")
        print("━" * 35 + "\n")
        tokenizer, text_tokens = prepare_tokenizer_and_dataset(cfg.vocab_size, cfg.id)
        total_tokens = len(text_tokens)
        print(f"Ready for training: {total_tokens} tokens\n")

        # Create PyTorch DataLoaders (wrapped with MpDeviceLoader in create_data_loaders)
        self.train_loader, self.val_loader = create_data_loaders(
            text_tokens, 
            cfg, 
            val_split=cfg.val_split
        )

    def train(self):
        print("TPU TRAINING")
        print("━" * 35 + "\n")
        
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("DataLoaders not initialized. Call tokenize() first.")

        # Initialize model (XLA compilation is automatic)
        print("Initializing GPT model...")
        self.model = GPT(cfg).to(device=device, dtype=cfg.dtype)
        self.optimizer = build_optimizer(self.model, cfg)

        self.model.train()    
        # Warmup XLA graph compilation
        warmup_xla_graph(self.model, cfg.batch_size, cfg.sequence_length, device)
        warmup_steps = max(50, int(cfg.total_steps * 0.15))

        # Precompute learning rate schedule (avoids device syncs on TPU)
        def get_lr_scale(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (cfg.total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        lr_schedule = [cfg.lr * get_lr_scale(i) for i in range(cfg.total_steps)]

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
        
        while step < cfg.total_steps:
            # Get next batch from persistent iterator
            try:
                x_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x_batch, y_batch = next(train_iter)
            
            # Ensure correct dtypes (MpDeviceLoader already places on device)
            x_batch = x_batch.to(dtype=torch.long)
            y_batch = y_batch.to(dtype=torch.long)
            
            batch_tokens = x_batch.numel()
            
            # Update learning rate from precomputed schedule
            base_opt = self.optimizer.base if cfg.sam else self.optimizer
            lr = lr_schedule[step]
            if step % 2 == 0:
                base_opt.param_groups[0]['lr'] = lr
            
            # Forward pass (XLA handles mixed precision natively with BF16)
            logits, _ = self.model(x_batch)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y_batch.flatten(),
                label_smoothing=cfg.label_smoothing
            )
            
            # Backward pass
            loss.backward()
            loss_buffer.append(loss.detach())
            
            # Gradient clipping (disabled on TPU by default)
            if cfg.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
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
                base_opt.zero_grad()
            
            # Mark XLA step (critical for TPU: finalizes the training step)
            xm.mark_step()
            
            tokens_processed += batch_tokens
            
            # Update EMA weights infrequently
            if cfg.use_ema and step % ema_update_steps == 0:
                if self.ema_state_dict is None:
                    self.ema_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    for k, v in self.model.state_dict().items():
                        self.ema_state_dict[k] = cfg.ema_decay * self.ema_state_dict[k] + (1 - cfg.ema_decay) * v
            
            # =========================================================
            # VALIDATION & CHECKPOINTING
            # =========================================================
            if (step + 1) % cfg.val_interval == 0:
                # TPU: Never switch modes (eval/train) - use torch.no_grad() instead
                # Switching modes causes XLA recompilation even with dropout=0
                
                # Swap weights with EMA if available
                if cfg.use_ema and self.ema_state_dict is not None:
                    original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.model.load_state_dict(self.ema_state_dict)
                
                # Validate on limited batches (avoid expensive XLA syncs)
                val_losses = []
                max_val_batches = 2
                val_batch_count = 0
                with torch.no_grad():
                    for x_vbatch, y_vbatch in self.val_loader:
                        if val_batch_count >= max_val_batches:
                            break
                        
                        x_vbatch = x_vbatch.to(dtype=torch.long)
                        y_vbatch = y_vbatch.to(dtype=torch.long)
                        logits, _ = self.model(x_vbatch)
                        val_loss_batch = F.cross_entropy(
                            logits.flatten(0, 1),
                            y_vbatch.flatten()
                        )
                        val_losses.append(val_loss_batch)
                        val_batch_count += 1
                
                # Mark XLA step after validation (batches all graphs together)
                xm.mark_step()
                
                # Compute average loss (single device sync)
                if val_losses:
                    val_loss = torch.stack(val_losses).mean().item()
                else:
                    val_loss = float('inf')
                
                # Restore original weights before saving
                if cfg.use_ema and self.ema_state_dict is not None:
                    self.model.load_state_dict(original_state)
                
                # Save best model checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs("model", exist_ok=True)
                    
                    # Save with xm.save for XLA-aware checkpoint
                    xm.save(self.model.state_dict(), f"model/best_model_{cfg.id}.pt")
                    if cfg.use_ema and self.ema_state_dict is not None:
                        xm.save(self.ema_state_dict, f"model/best_ema_model_{cfg.id}.pt")
                
                elapsed = time.perf_counter() - start_time
                train_loss = torch.stack(loss_buffer).mean().item() if loss_buffer else 0.0
                loss_buffer.clear()
                throughput = tokens_processed / elapsed if elapsed > 0 else 0
                print(f"Step {step+1:5d} | Train {train_loss:.4f} | Val {val_loss:.4f} | {throughput:.0f} tok/s")
            
            step += 1


if __name__ == "__main__":
    control = Control()
    control.display()
    control.tokenize()
    control.train()