# core.py (TPU)

import torch
from utils import activation_fn
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# TPU import (required)
import torch_xla.core.xla_model as xm


def compile_model(model: nn.Module, mode: str = 'reduce-overhead') -> nn.Module:
    """No-op on TPU (XLA compilation is automatic)."""
    return model

# =========================================================
# TOKENIZER
# =========================================================

class BPETokenizer:
    """BPE tokenizer (Rust-based HuggingFace) with special tokens and compression tracking."""
    
    def __init__(self, vocab_size: int, min_frequency: int = 2, tokenizer_id: str = "default"):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer_id = tokenizer_id
        self.pad_id = self.unk_id = self.bos_id = self.eos_id = None
        
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, text, show_stats=True):
        """Train tokenizer on text and cache special token IDs."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min(self.min_frequency, 5),
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            show_progress=True,
            limit_alphabet=1000,
            initial_alphabet=[chr(i) for i in range(256)],
        )
        
        # Train on chunks for better memory usage and merge quality
        chunk_size = 16_777_216
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
        """Encode text to token IDs, optionally adding BOS/EOS."""
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
        """Decode token IDs to text, optionally filtering special tokens."""
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            special_ids.discard(None)
            ids = [id_ for id_ in ids if id_ not in special_ids]
        
        # Use HuggingFace decoder for proper ByteLevel decoding
        text = self.tokenizer.decode(ids)
        return text

    def decode_batch(self, batch_ids, skip_special_tokens=True):
        """Decode multiple sequences efficiently."""
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            special_ids.discard(None)  # Remove None if any IDs weren't resolved
            batch_ids = [
                [id_ for id_ in ids if id_ not in special_ids]
                for ids in batch_ids
            ]
        return self.tokenizer.decode_batch(batch_ids)

    def get_vocab_size(self):
        """Get actual vocabulary size."""
        return len(self.tokenizer.get_vocab())

    def get_compression_ratio(self, text):
        """Return bytes-per-token ratio (higher = better compression)."""
        num_tokens = len(self.encode(text))
        return len(text) / max(num_tokens, 1)

    def save(self, filepath):
        """Save tokenizer to disk."""
        self.tokenizer.save(filepath)
        print(f"Saved tokenizer '{self.tokenizer_id}' to {filepath}")

    def load(self, filepath):
        """Load tokenizer from disk and cache special token IDs."""
        self.tokenizer = Tokenizer.from_file(filepath)
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        print(f"Loaded tokenizer '{self.tokenizer_id}' from {filepath}")

    def _print_vocab_stats(self):
        """Print vocabulary statistics."""
        actual_size = len(self.tokenizer.get_vocab())
        print(f"\nTokenizer [{self.tokenizer_id}]: {actual_size} / {self.vocab_size} tokens")
        print(f"  Special: pad={self.pad_id}, unk={self.unk_id}, bos={self.bos_id}, eos={self.eos_id}")


# =========================================================
# RMSNorm (RMS Layer Normalization)
# =========================================================
class RMSNorm(nn.Module):
    """RMS Layer Normalization (PyTorch ≥2.1)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


# =========================================================
# RoPE
# =========================================================
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, yarn_scale=1.0):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.yarn_scale = yarn_scale
        
        # Precompute sin/cos caches for all positions
        rope_max_len = max(max_seq_len, int(max_seq_len * yarn_scale))
        freqs = torch.outer(torch.arange(rope_max_len, dtype=torch.float32), inv_freq)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.rope_max_len = rope_max_len
        
        # Precompute position indices to avoid torch.arange() in forward pass
        pos_arange = torch.arange(rope_max_len, dtype=torch.float32)
        scaled_indices = torch.clamp(pos_arange / yarn_scale, 0, rope_max_len - 1).long()
        self.register_buffer("scaled_indices", scaled_indices, persistent=False)
        self.register_buffer("unscaled_indices", torch.arange(rope_max_len, dtype=torch.long), persistent=False)

    def _apply_rope(self, x, indices):
        """Core RoPE: rotate using precomputed sin/cos and position indices."""
        sin = self.sin[indices].unsqueeze(0).unsqueeze(2).to(x.dtype)
        cos = self.cos[indices].unsqueeze(0).unsqueeze(2).to(x.dtype)
        
        # Interleave and rotate (stack+flatten avoids torch.empty_like overhead)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
            dim=-1
        ).flatten(-2)
        return rotated

    def forward_train(self, x, start_pos=0):
        """Training RoPE (no YaRN scaling). XLA compiles as separate unbranched graph."""
        T = x.size(1)
        indices = self.unscaled_indices[start_pos:start_pos + T]
        return self._apply_rope(x, indices)

    def forward_infer(self, x, start_pos=0):
        """Inference RoPE (YaRN scaling). XLA compiles as separate unbranched graph."""
        T = x.size(1)
        indices = self.scaled_indices[start_pos:start_pos + T]
        return self._apply_rope(x, indices)

    def forward(self, x, start_pos=0, inference=False):
        """Python-time dispatch to training or inference RoPE path (zero HLO branching)."""
        if inference:
            return self.forward_infer(x, start_pos)
        else:
            return self.forward_train(x, start_pos)


# =========================================================
# ATTENTION (Full MHA for TPU)
# =========================================================
class Attention(nn.Module):
    """Full Multi-Head Attention (no KV reduction) optimized for TPU."""
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads  # No KV reduction on TPU
        self.head_dim = cfg.d_model // cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0
        
        # QKV projection (receives pre-normalized input from Block)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.n_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.proj_drop = nn.Dropout(cfg.dropout)
        
        yarn_scale = getattr(cfg, 'yarn_scale', 1.0)
        self.rope = RoPE(self.head_dim, max_seq_len=cfg.sequence_length, yarn_scale=yarn_scale)

    def forward(self, x_norm, cache=None):
        """Attention forward with pre-normalized input and optional KV cache."""
        B, T, C = x_norm.shape
        pos = cache[2] if cache is not None else 0
        inference = cache is not None

        # QKV projection on pre-normalized input
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        # Specialized RoPE paths (no branching in compiled graph)
        if inference:
            q = self.rope.forward_infer(q, start_pos=pos)
            k = self.rope.forward_infer(k, start_pos=pos)
        else:
            q = self.rope.forward_train(q, start_pos=pos)
            k = self.rope.forward_train(k, start_pos=pos)
        
        # Handle cache
        if cache is not None:
            k_cache, v_cache, _ = cache
            max_pos = k_cache.shape[1]
            assert pos + T <= max_pos, f"KV cache overflow: pos={pos}, T={T}, max={max_pos}"
            
            from torch_xla.core.functions import dynamic_update_slice
            k_cache = dynamic_update_slice(k_cache, k, (0, pos, 0, 0))
            v_cache = dynamic_update_slice(v_cache, v, (0, pos, 0, 0))
            
            k, v = k_cache, v_cache
            new_pos = pos + T
        else:
            new_pos = None
        
        # Transpose to (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA (full MHA on TPU - no GQA)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.proj_drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        
        if cache is not None:
            return self.proj_drop(self.out_proj(out)), (k_cache, v_cache, new_pos)
        else:
            return self.proj_drop(self.out_proj(out)), None

# =========================================================
# FUSED MLP (gate+value projected together, fused by torch.compile)
# =========================================================
class FusedMLP(nn.Module):
    """SwiGLU MLP with fused RMSNorm+FC projection."""
    def __init__(self, d_model, hidden_dim, activation="swiglu"):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, 2 * hidden_dim, bias=True)
        self.proj = nn.Linear(hidden_dim, d_model, bias=True)
        self.activation = activation
        self.act = (
            activation_fn(activation)
            if activation not in ("swiglu", "geglu", "reglu")
            else None
        )
    
    def forward(self, x):
        """Fuse RMSNorm + FC projection, then gate-value activation."""
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.ln.eps) * self.ln.weight
        gv = F.linear(x_norm, self.fc.weight, self.fc.bias)
        u, v = gv.chunk(2, dim=-1)
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
    """Transformer block: pre-norm + attention + residual + pre-norm + MLP + residual."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        # 64-align hidden dim for TPU systolic array efficiency
        hidden = int(4/3 * cfg.mlp_ratio * cfg.d_model)
        hidden = (hidden + 63) // 64 * 64
        self.mlp = FusedMLP(cfg.d_model, hidden, cfg.activation)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        """Pre-norm residual blocks with improved torch.compile fusion."""
        # Attention: pre-norm + attn + residual + dropout
        if self.gradient_checkpointing and self.training:
            def attn_fn(x_):
                return self.attn(self.ln1(x_), cache)
            attn_out, cache = torch.utils.checkpoint.checkpoint(
                attn_fn, x, use_reentrant=False
            )
        else:
            attn_out, cache = self.attn(self.ln1(x), cache)
        x = x + self.drop1(attn_out)

        # MLP: pre-norm + mlp + residual + dropout
        if self.gradient_checkpointing and self.training:
            def mlp_fn(x_):
                return self.mlp(self.ln2(x_))
            mlp_out = torch.utils.checkpoint.checkpoint(
                mlp_fn, x, use_reentrant=False
            )
        else:
            mlp_out = self.mlp(self.ln2(x))
        x = x + self.drop2(mlp_out)
        
        return x, cache

# =========================================================
# GPT MODEL
# =========================================================
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            # Scale init std by 1/sqrt(n_layers) for deeper networks
            std = self.cfg.init_std / (self.cfg.n_layers ** 0.5)
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def allocate_kv_cache(self, batch_size, max_seq_len, device, dtype):
        """Preallocate full MHA KV cache for generation."""
        cache = []
        head_dim = self.cfg.d_model // self.cfg.n_heads
        
        for _ in range(self.cfg.n_layers):
            k_cache = torch.zeros(batch_size, max_seq_len, self.cfg.n_heads, head_dim,
                                 device=device, dtype=dtype)
            v_cache = torch.zeros_like(k_cache)
            cache.append((k_cache, v_cache, 0))
        return cache

    def forward(self, idx, cache=None):
        """Forward with optional KV cache. idx shape: (B, T), returns logits (B, T, vocab_size)."""
        x = self.tok(idx)
        
        if cache is None:
            for blk in self.blocks:
                x, _ = blk(x, None)
            return self.head(self.ln_f(x)), None
        
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, blk_cache = blk(x, cache[i])
            new_cache.append(blk_cache)
        return self.head(self.ln_f(x)), new_cache

    @torch.no_grad()
    def forward_prefill(self, idx, cache):
        """Prefill phase: process prompt (B, T) with T > 1. Separate XLA compilation from decode."""
        x = self.tok(idx)
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, blk_cache = blk(x, cache[i])
            new_cache.append(blk_cache)
        logits = self.head(self.ln_f(x))
        xm.mark_step()  # Finalize prefill computation
        return logits, new_cache

    @torch.no_grad()
    def forward_generate(self, idx, cache, step):
        """Generation forward with static T=1 for XLA (separate compiled graph)."""
        assert idx.shape[1] == 1, f"forward_generate requires T=1, got T={idx.shape[1]}"
        
        x = self.tok(idx)
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, blk_cache = blk(x, cache[i])
            new_cache.append(blk_cache)
        
        logits = self.head(self.ln_f(x))
        xm.mark_step()  # Finalize XLA computation and flush to device
        return logits, new_cache