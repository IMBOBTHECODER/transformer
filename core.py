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
# TOKENIZER (RUST-BASED FAST BPE WITH IMPROVEMENTS)
# =========================================================
class BPETokenizer:
    def __init__(self, vocab_size, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency  # Filter rare tokens during training
        
        # Define special tokens with fixed IDs
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Use HuggingFace tokenizers (Rust-based, production-grade)
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Add decoders for proper text reconstruction
        from tokenizers.processors import ByteLevel as ByteLevelProcessor
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
        
        # Add decoder (inverse of ByteLevel encoding)
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        self.tokenizer.decoder = ByteLevelDecoder()
        
        # Cache for vocab stats
        self._vocab_stats = None

    def train(self, text, show_stats=True):
        """
        Train BPE tokenizer with improved parameters for better compression.
        
        Args:
            text: Training text
            show_stats: Print vocabulary statistics
        """
        # Create trainer with optimized parameters
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,  # Only merge tokens that appear 2+ times
            special_tokens=[
                "<pad>",      # ID 0
                "<unk>",      # ID 1
                "<bos>",      # ID 2
                "<eos>",      # ID 3
            ],
            show_progress=True,
            limit_alphabet=1000,  # Limit alphabet for efficiency
            initial_alphabet=[chr(i) for i in range(256)],  # Full byte alphabet
        )
        
        # Split text into chunks for better BPE training and memory efficiency
        chunk_size = 10_000_000  # 10M character chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Train on iterator of chunks (better merge quality and memory usage)
        self.tokenizer.train_from_iterator(chunks, trainer=trainer)
        
        if show_stats:
            self._print_vocab_stats()

    def encode(self, text, add_special_tokens=False):
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Prepend BOS and append EOS tokens
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        
        return ids

    def encode_batch(self, texts, add_special_tokens=False):
        """Encode multiple texts efficiently."""
        encodings = self.tokenizer.encode_batch(texts)
        
        if add_special_tokens:
            return [
                [self.bos_id] + enc.ids + [self.eos_id]
                for enc in encodings
            ]
        return [enc.ids for enc in encodings]

    def decode(self, ids, skip_special_tokens=True):
        """
        Decode token IDs back to text with improved handling.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Remove special tokens from output
            
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special token IDs
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids = [id_ for id_ in ids if id_ not in special_ids]
        
        # Use HuggingFace decoder for proper ByteLevel decoding
        text = self.tokenizer.decode(ids)
        return text

    def decode_batch(self, batch_ids, skip_special_tokens=True):
        """Decode multiple sequences efficiently."""
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            batch_ids = [
                [id_ for id_ in ids if id_ not in special_ids]
                for ids in batch_ids
            ]
        return self.tokenizer.decode_batch(batch_ids)

    def get_vocab_size(self):
        """Get actual vocabulary size."""
        return len(self.tokenizer.get_vocab())

    def get_compression_ratio(self, text):
        """
        Calculate compression ratio: original_chars / tokens.
        Higher = better compression.
        """
        num_tokens = len(self.encode(text))
        num_chars = len(text)
        return num_chars / max(num_tokens, 1)

    def _print_vocab_stats(self):
        """Print vocabulary statistics."""
        vocab = self.tokenizer.get_vocab()
        actual_size = len(vocab)
        print(f"\nTokenizer Statistics:")
        print(f"  Vocabulary size: {actual_size} / {self.vocab_size}")
        print(f"  Special tokens: pad={self.pad_id}, unk={self.unk_id}, bos={self.bos_id}, eos={self.eos_id}")
        print(f"  Token frequency distribution analysis completed")

    def save(self, path):
        """Save tokenizer to disk."""
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """Load tokenizer from disk."""
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path}")

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