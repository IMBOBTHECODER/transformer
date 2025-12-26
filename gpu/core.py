# core.py (GPU-only)
"""
GPU-optimized transformer model with flash attention, GQA, and mixed precision support.
Uses torch.compile for automatic kernel fusion and optimization.
"""

import torch
from utils import activation_fn
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def compile_model(model, mode='reduce-overhead'):
    """Compile model with torch.compile for kernel fusion and faster inference."""
    return torch.compile(model, mode=mode)

# =========================================================
# TOKENIZER
# =========================================================
class BPETokenizer:
    """BPE tokenizer using HuggingFace tokenizers (Rust-based, production-grade)."""
    
    def __init__(self, vocab_size, min_frequency=2, tokenizer_id="default"):
        self.tokenizer_id = tokenizer_id
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special token IDs (resolved after training)
        self.pad_id = self.unk_id = self.bos_id = self.eos_id = None
        
        # HuggingFace tokenizer with byte-level encoding
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, text, show_stats=True):
        """Train BPE tokenizer on text."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min(self.min_frequency, 5),
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            show_progress=True,
            limit_alphabet=1000,
            initial_alphabet=[chr(i) for i in range(256)],
        )
        
        # Split into chunks for efficient training
        chunk_size = 16_777_216  # 16MB chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        self.tokenizer.train_from_iterator(chunks, trainer=trainer)
        
        # Query actual special token IDs from trained tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        
        if show_stats:
            self._print_vocab_stats()

    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        
        if add_special_tokens:
            if self.bos_id is not None:
                ids = [self.bos_id] + ids
            if self.eos_id is not None:
                ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts, add_special_tokens=False):
        """Encode multiple texts efficiently."""
        encodings = self.tokenizer.encode_batch(texts)
        
        if add_special_tokens:
            result = []
            for enc in encodings:
                ids = enc.ids
                if self.bos_id is not None:
                    ids = [self.bos_id] + ids
                if self.eos_id is not None:
                    ids = ids + [self.eos_id]
                result.append(ids)
            return result
        return [enc.ids for enc in encodings]

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            special_ids.discard(None)
            ids = [id_ for id_ in ids if id_ not in special_ids]
        return self.tokenizer.decode(ids)

    def decode_batch(self, batch_ids, skip_special_tokens=True):
        """Decode multiple sequences efficiently."""
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            special_ids.discard(None)
            batch_ids = [
                [id_ for id_ in ids if id_ not in special_ids]
                for ids in batch_ids
            ]
        return self.tokenizer.decode_batch(batch_ids)

    def get_vocab_size(self):
        """Get actual vocabulary size."""
        return len(self.tokenizer.get_vocab())

    def get_compression_ratio(self, text):
        """Calculate compression ratio (original_chars / tokens). Higher is better."""
        num_tokens = len(self.encode(text))
        return len(text) / max(num_tokens, 1)

    def save(self, filepath):
        """Save tokenizer to disk."""
        self.tokenizer.save(filepath)
        print(f"Saved tokenizer '{self.tokenizer_id}' to {filepath}")

    def load(self, filepath):
        """Load tokenizer from disk and resolve special token IDs."""
        self.tokenizer = Tokenizer.from_file(filepath)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        print(f"Loaded tokenizer '{self.tokenizer_id}' from {filepath}")

    def _print_vocab_stats(self):
        """Print vocabulary statistics."""
        vocab = self.tokenizer.get_vocab()
        print(f"\nTokenizer Statistics [{self.tokenizer_id}]:")
        print(f"  Vocabulary size: {len(vocab)} / {self.vocab_size}")
        print(f"  Special tokens: pad={self.pad_id}, unk={self.unk_id}, bos={self.bos_id}, eos={self.eos_id}")


# =========================================================
# RoPE (Rotary Position Embeddings)
# =========================================================
class RoPE(nn.Module):
    """Rotary position embeddings with YaRN scaling for extended context."""
    
    def __init__(self, head_dim, max_seq_len=2048, yarn_scale=1.0):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # YaRN: compress positions for extended context during generation
        self.yarn_scale = yarn_scale
        rope_max_len = max(max_seq_len, int(max_seq_len * yarn_scale))
        
        # Precompute sin/cos for all positions (avoid per-token computation)
        freqs = torch.outer(torch.arange(rope_max_len, dtype=torch.float32), inv_freq)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.rope_max_len = rope_max_len

    def forward(self, x, start_pos=0, inference=False):
        """Apply RoPE with optional YaRN scaling for extended context."""
        T = x.size(1)
        
        if inference and self.yarn_scale > 1.0:
            # Scale positions down for extended context during generation
            scaled_pos = torch.arange(start_pos, start_pos + T, dtype=torch.float32, device=x.device) / self.yarn_scale
            scaled_pos = torch.clamp(scaled_pos, 0, self.rope_max_len - 1)
            indices = scaled_pos.long()
            sin_cache = self.sin[indices]
            cos_cache = self.cos[indices]
        else:
            # Use precomputed cache (faster for training)
            sin_cache = self.sin[start_pos:start_pos+T]
            cos_cache = self.cos[start_pos:start_pos+T]
        
        sin = sin_cache.unsqueeze(0).unsqueeze(2).to(x.dtype)
        cos = cos_cache.unsqueeze(0).unsqueeze(2).to(x.dtype)
        
        # Apply rotation: [x_odd, x_even] -> [x_odd*cos - x_even*sin, x_odd*sin + x_even*cos]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated


# =========================================================
# RMSNorm (Root Mean Square Layer Normalization)
# =========================================================
class RMSNorm(nn.Module):
    """RMS normalization using PyTorch's built-in (v2.1+). Simpler than LayerNorm."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
# =========================================================
# ATTENTION (Flash Attention + Grouped Query Attention)
# =========================================================
class Attention(nn.Module):
    """Multi-head attention with GQA for memory efficiency and flash attention."""
    
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads // 2  # GQA: 4 KV heads vs 8 Q heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // self.n_kv_heads
        
        # Separate projections for GQA
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=True)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        
        # RoPE for position embeddings
        yarn_scale = getattr(cfg, 'yarn_scale', 1.0)
        self.rope = RoPE(self.head_dim, max_seq_len=cfg.sequence_length, yarn_scale=yarn_scale)

    def forward(self, x, cache=None):
        """Forward pass with optional KV cache for generation."""
        B, T, C = x.shape
        pos = cache[2] if cache is not None else 0
        inference = cache is not None

        # Project Q and KV
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        kv = self.kv_proj(x).view(B, T, 2 * self.n_kv_heads, self.head_dim)
        k, v = kv[..., :self.n_kv_heads, :], kv[..., self.n_kv_heads:, :]
        
        # Apply RoPE before transpose
        q = self.rope(q, start_pos=pos, inference=inference)
        k = self.rope(k, start_pos=pos, inference=inference)
        
        # Handle KV cache
        if cache is not None:
            k_cache, v_cache, _ = cache
            max_pos = k_cache.shape[1]
            assert pos + T <= max_pos, f"KV cache overflow: pos={pos}, T={T}, max={max_pos}"
            k_cache[:, pos:pos+T, :, :] = k
            v_cache[:, pos:pos+T, :, :] = v
            k, v = k_cache, v_cache
            new_pos = pos + T
        else:
            new_pos = None
        
        # Transpose to (B, H, T, D) for flash attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV for GQA (repeat KV heads to match Q)
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Flash attention: optimized CUDA kernel
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.proj_drop.p if self.training else 0.0
        )
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj_drop(self.out_proj(out))
        
        if cache is not None:
            return out, (k_cache, v_cache, new_pos)
        return out, None

# =========================================================
# Fused MLP (Feed-Forward Network)
# =========================================================
class FusedMLP(nn.Module):
    """Fused MLP: Linear(2x) → chunk → activation → multiply → Linear. torch.compile optimizes this."""
    
    def __init__(self, d_model, hidden_dim, activation="swiglu"):
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, d_model, bias=True)
        self.activation = activation
        self.act = (
            activation_fn(activation)
            if activation not in ("swiglu", "geglu", "reglu")
            else None
        )
    
    def forward(self, x):
        """Fused forward: torch.compile merges all ops into minimal kernels."""
        u, v = self.fc(x).chunk(2, dim=-1)
        if self.activation == "swiglu":
            return self.proj(F.silu(u) * v)
        elif self.activation == "geglu":
            return self.proj(F.gelu(u) * v)
        elif self.activation == "reglu":
            return self.proj(F.relu(u) * v)
        else:
            return self.proj(self.act(u) * v)

# =========================================================
# TRANSFORMER BLOCK
# =========================================================
class Block(nn.Module):
    """Transformer block: attention → MLP with residual connections and layer norm."""
    
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

        # Align hidden dim to GPU tensor cores (multiples of 64)
        hidden_dim = ((int(4/3 * cfg.mlp_ratio * cfg.d_model) + 63) // 64) * 64
        self.mlp = FusedMLP(cfg.d_model, hidden_dim, cfg.activation)

    def forward(self, x, cache=None):
        # Attention with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def attn_fn(x_):
                return self.attn(self.ln1(x_), cache)
            a, cache = torch.utils.checkpoint.checkpoint(attn_fn, x, use_reentrant=False)
        else:
            a, cache = self.attn(self.ln1(x), cache)
        x = x + self.drop1(a)

        # MLP with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def mlp_fn(x_):
                return self.mlp(self.ln2(x_))
            h = torch.utils.checkpoint.checkpoint(mlp_fn, x, use_reentrant=False)
        else:
            h = self.mlp(self.ln2(x))
        
        x = x + self.drop2(h)
        return x, cache

# =========================================================
# GPT MODEL
# =========================================================
class GPT(nn.Module):
    """GPT-style transformer with flash attention and GQA for efficient training/inference."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # Weight tying
        self.apply(self._init)

    def _init(self, m):
        """Initialize weights with scaled std for deeper networks."""
        if isinstance(m, nn.Linear):
            std = self.cfg.init_std / (self.cfg.n_layers ** 0.5)
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def allocate_kv_cache(self, batch_size, max_seq_len, device, dtype):
        """Preallocate KV cache for efficient generation."""
        cache = []
        n_kv_heads = self.cfg.n_heads // 2  # GQA
        head_dim = self.cfg.d_model // self.cfg.n_heads
        
        for _ in range(self.cfg.n_layers):
            k_cache = torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim,
                                 device=device, dtype=dtype)
            v_cache = torch.zeros_like(k_cache)
            cache.append((k_cache, v_cache, 0))
        return cache

    def forward(self, idx, cache=None):
        """Forward pass with optional KV cache for generation."""
        x = self.tok(idx)
        
        if cache is None:
            # Training or prefill: no cache
            for blk in self.blocks:
                x, _ = blk(x, None)
        else:
            # Generation: use KV cache
            new_cache = []
            for i, blk in enumerate(self.blocks):
                x, blk_cache = blk(x, cache[i])
                new_cache.append(blk_cache)
            cache = new_cache

        logits = self.head(self.ln_f(x))
        return logits, cache