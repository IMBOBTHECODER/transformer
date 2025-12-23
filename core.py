# core.py
import torch
from utils import activation_fn
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# Try to import Flash Attention, fall back gracefully
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None

# =========================================================
# TOKENIZER (RUST-BASED FAST BPE)
# =========================================================
class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
        # Define special tokens with fixed IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Use HuggingFace tokenizers (Rust-based, production-grade)
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    def train(self, text):
        """Train BPE tokenizer on text using Rust implementation."""
        # Create trainer with explicit special tokens
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[
                "<pad>",      # ID 0
                "<unk>",      # ID 1
                "<bos>",      # ID 2
                "<eos>",      # ID 3
            ],
            show_progress=False
        )
        
        # Split text into chunks for better BPE training and memory efficiency
        chunk_size = 10_000_000  # 10M character chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Train on iterator of chunks (better merge quality and memory usage)
        self.tokenizer.train_from_iterator(chunks, trainer=trainer)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # Verify special token IDs match our expectations
        assert self.tokenizer.token_to_id("<pad>") == self.pad_id, "Padding ID mismatch"
        assert self.tokenizer.token_to_id("<unk>") == self.unk_id, "Unknown ID mismatch"
        assert self.tokenizer.token_to_id("<bos>") == self.bos_id, "BOS ID mismatch"
        assert self.tokenizer.token_to_id("<eos>") == self.eos_id, "EOS ID mismatch"

    def encode(self, text):
        """Encode text to token IDs (ultra-fast Rust implementation)."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        text = self.tokenizer.decode(ids)
        # Clean up ByteLevel special characters
        text = text.replace('Ġ', ' ')  # Replace space token
        text = text.replace('Ċ', '\n')  # Replace newline token
        # Remove multiple consecutive spaces (but keep newlines)
        import re
        text = re.sub(r' {2,}', ' ', text)
        # Remove spaces before punctuation (common post-processing)
        text = re.sub(r' ([.,!?;:\'\"\)\]}])', r'\1', text)
        return text

# =========================================================
# RoPE
# =========================================================
class RoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._sin_cos_cache = {}

    def forward(self, x):
        # x: (B, H, T, D)
        T = x.size(2)
        cache_key = (T, x.device, x.dtype)
        
        # Use cache for sin/cos if available (inference only - reduces memory during training)
        if cache_key in self._sin_cos_cache:
            sin, cos = self._sin_cos_cache[cache_key]
        else:
            inv_freq = self.inv_freq.to(x.dtype)
            freqs = torch.outer(
                torch.arange(T, device=x.device, dtype=x.dtype),
                inv_freq
            )
            sin, cos = freqs.sin(), freqs.cos()
            # Cache results only during inference
            if not self.training and len(self._sin_cos_cache) < 10:
                self._sin_cos_cache[cache_key] = (sin, cos)

        # Optimized RoPE: reshape sin/cos for efficient broadcasting
        # (T, D/2) -> (1, 1, T, D/2) for proper broadcasting with (B, H, T, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        
        # Apply RoPE with in-place operations for memory efficiency
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1
        )

# =========================================================
# ATTENTION (FLASH + KV)
# =========================================================
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.use_flash = cfg.use_flash_attention
        
        # Fused QKV projection: single linear layer for all three
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        
        # Use dropout from config instead of hardcoded fallback
        dropout = cfg.dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = RoPE(self.head_dim)
        self.has_flash = HAS_FLASH_ATTN

    def forward(self, x, cache=None):
        B, T, C = x.shape

        # Fused QKV projection: single matrix mult for all three
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Split into Q, K, V
        
        # Reshape to (B, H, T, D) after splitting
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Fuse RoPE: apply immediately after projection before attention
        q = self.rope(q)  # RoPE on Q
        k = self.rope(k)  # RoPE on K

        # Handle cache with proper dtype
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache = {"k": k.detach(), "v": v.detach()}

        # Use Flash Attention if available and enabled, else fall back to scaled_dot_product_attention
        if self.use_flash and self.has_flash:
            # Flash Attention: 2-3x faster, requires (B, T, H, D) format
            out = flash_attn_func(q, k, v, causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = self.attn_drop(out)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.out_proj(out)), cache

# =========================================================
# TRANSFORMER BLOCK
# =========================================================
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        
        # Use dropout from config
        dropout = cfg.dropout
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        hidden = cfg.mlp_ratio * cfg.d_model
        # Use bias for better expressiveness
        self.fc = nn.Linear(cfg.d_model, hidden * 2, bias=True)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=True)

        self.activation = cfg.activation
        self.act = (
            activation_fn(cfg.activation)
            if cfg.activation not in ("swiglu", "geglu", "reglu")
            else None
        )

    def forward(self, x, cache=None):
        # Attention block with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def attn_forward(x_):
                return self.attn(self.ln1(x_), cache)
            a, cache = torch.utils.checkpoint.checkpoint(
                attn_forward, x, use_reentrant=False
            )
        else:
            a, cache = self.attn(self.ln1(x), cache)
        x = x + self.drop1(a)

        # Fused SwiGLU / gated activation with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def mlp_forward(x_):
                h_input = self.ln2(x_)
                u, v = self.fc(h_input).chunk(2, dim=-1)
                if self.activation == "swiglu":
                    h = F.silu(u) * v
                elif self.activation == "geglu":
                    h = F.gelu(u) * v
                elif self.activation == "reglu":
                    h = F.relu(u) * v
                else:
                    h = self.act(u)
                return self.proj(h)
            h = torch.utils.checkpoint.checkpoint(
                mlp_forward, x, use_reentrant=False
            )
        else:
            h_input = self.ln2(x)
            u, v = self.fc(h_input).chunk(2, dim=-1)
            if self.activation == "swiglu":
                h = F.silu(u) * v
            elif self.activation == "geglu":
                h = F.gelu(u) * v
            elif self.activation == "reglu":
                h = F.relu(u) * v
            else:
                h = self.act(u)
            h = self.proj(h)

        x = x + self.drop2(h)
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
        self.ln_f = nn.LayerNorm(cfg.d_model)
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

    def forward(self, idx, cache=None):
        x = self.tok(idx)
        
        if cache is None:
            for blk in self.blocks:
                x, _ = blk(x, None)
            return self.head(self.ln_f(x)), None
        
        # With cache
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, blk_cache = blk(x, cache[i])
            new_cache.append(blk_cache)

        return self.head(self.ln_f(x)), new_cache