# core.py
import torch
from utils import activation_fn
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# =========================================================
# TOKENIZER (RUST-BASED FAST BPE WITH IMPROVEMENTS)
# =========================================================
class BPETokenizer:
    def __init__(self, vocab_size, min_frequency=2, tokenizer_id="default"):
        self.tokenizer_id = tokenizer_id  # Identifier for this tokenizer
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
            min_frequency=min(self.min_frequency, 5),  # Increase to 5-10 range for better compression
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
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        
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

    def save(self, filepath):
        """Save tokenizer to disk."""
        self.tokenizer.save(filepath)
        print(f"Tokenizer '{self.tokenizer_id}' saved to {filepath}")

    def load(self, filepath):
        """Load tokenizer from disk and preserve tokenizer_id."""
        self.tokenizer = Tokenizer.from_file(filepath)
        print(f"Tokenizer '{self.tokenizer_id}' loaded from {filepath}")

    def _print_vocab_stats(self):
        """Print vocabulary statistics."""
        vocab = self.tokenizer.get_vocab()
        actual_size = len(vocab)
        print(f"\nTokenizer Statistics [{self.tokenizer_id}]:")
        print(f"  Vocabulary size: {actual_size} / {self.vocab_size}")
        print(f"  Special tokens: pad={self.pad_id}, unk={self.unk_id}, bos={self.bos_id}, eos={self.eos_id}")
        print(f"  Token frequency distribution analysis completed")

# =========================================================
# RMSNorm (RMS Layer Normalization)
# =========================================================
class RMSNorm(nn.Module):
    """RMS Layer Normalization using PyTorch's built-in (≥2.1)."""
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
        
        # YaRN: extend context length beyond training
        self.yarn_scale = yarn_scale
        
        # Precompute sin/cos for all positions (cache as buffers, not dict)
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, x, start_pos=0, inference=False):
        # x: (B, T, H, D) - applies RoPE with absolute positions for correct cache alignment
        T = x.size(1)
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.float32, device=x.device)
        
        # YaRN: inference-only frequency interpolation for extended context during generation
        if inference and self.yarn_scale > 1.0:
            # Smooth interpolation: blend old and new frequencies (inference only)
            t = pos / self.yarn_scale
            t = torch.clamp(t, 0, pos.max())  # Avoid out-of-bounds
            sin_cache = torch.sin(torch.outer(t, self.inv_freq))
            cos_cache = torch.cos(torch.outer(t, self.inv_freq))
        else:
            # Use precomputed buffers for training (faster, always used during training)
            sin_cache = self.sin[start_pos:start_pos+T]
            cos_cache = self.cos[start_pos:start_pos+T]
        
        # Cast to input dtype and reshape for broadcasting
        sin = sin_cache.unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, T, D/2)
        cos = cos_cache.unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, T, D/2)
        
        # Apply RoPE: non-in-place for autograd safety
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos
        return rotated

# =========================================================
# ATTENTION (FLASH + KV)
# =========================================================
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads // 2
        self.head_dim = cfg.d_model // cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = cfg.n_heads // self.n_kv_heads  # Repeat factor for KV
        
        # Separate Q and KV projections for GQA
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=True)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        # YaRN: use yarn_scale > 1.0 for longer context during generation
        yarn_scale = getattr(cfg, 'yarn_scale', 1.0)
        self.rope = RoPE(self.head_dim, max_seq_len=cfg.sequence_length, yarn_scale=yarn_scale)

    def forward(self, x, cache=None):
        B, T, C = x.shape

        # Get absolute position from cache (0 if no cache)
        pos = cache[2] if cache is not None else 0
        inference = cache is not None  # YaRN only during inference

        # Separate Q and KV projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)  # (B, T, n_heads, D)
        kv = self.kv_proj(x).view(B, T, 2 * self.n_kv_heads, self.head_dim)
        k, v = kv[..., :self.n_kv_heads, :], kv[..., self.n_kv_heads:, :]  # (B, T, n_kv_heads, D)
        
        # Apply RoPE with absolute positions BEFORE transpose (more efficient)
        q = self.rope(q, start_pos=pos, inference=inference)
        k = self.rope(k, start_pos=pos, inference=inference)
        
        # Handle cache: keep in (B, T, H_kv, D) format for efficiency
        if cache is not None:
            k_cache, v_cache, _ = cache
            k_cache[:, pos:pos+T, :, :] = k  # k in (B, T, H_kv, D)
            v_cache[:, pos:pos+T, :, :] = v
            k, v = k_cache, v_cache
            new_pos = pos + T
        else:
            new_pos = None
        
        # Transpose to (B, H, T, D) only for SDPA
        q = q.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)
        k = k.transpose(1, 2)  # (B, T, H_kv, D) -> (B, H_kv, T, D)
        v = v.transpose(1, 2)  # (B, T, H_kv, D) -> (B, H_kv, T, D)

        # Use PyTorch SDPA with built-in GQA broadcasting
        # SDPA automatically broadcasts (B, H_kv, T, D) KV to match (B, H, T, D) Q
        out = F.scaled_dot_product_attention(
            q, k, v,  # k,v stay as (B, H_kv, T, D), q is (B, H, T, D)
            is_causal=True,
            dropout_p=self.proj_drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.out_proj(out)), (k_cache, v_cache, new_pos) if cache is not None else None

# =========================================================
# FUSED MLP (gate+value projected together, fused by torch.compile)
# =========================================================
class FusedMLP(nn.Module):
    """Fused MLP: Linear(2x) → chunk → activation → multiply → Linear (optimized by torch.compile)."""
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
        """Single fused forward for torch.compile optimization."""
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
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.ln2 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        
        dropout = cfg.dropout
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # Align hidden dim to tensor cores (multiples of 64)
        def round_to(x, m=64):
            return (x + m - 1) // m * m
        
        hidden_dim = round_to(int(2/3 * cfg.mlp_ratio * cfg.d_model))
        # Fused MLP: Linear(2x) → chunk → activation → multiply → Linear
        # torch.compile fuses all operations into minimal kernels
        self.mlp = FusedMLP(cfg.d_model, hidden_dim, cfg.activation)

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

        # Fused MLP with gradient checkpointing
        if self.gradient_checkpointing and self.training:
            def mlp_forward(x_):
                return self.mlp(self.ln2(x_))
            h = torch.utils.checkpoint.checkpoint(
                mlp_forward, x, use_reentrant=False
            )
        else:
            h = self.mlp(self.ln2(x))

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
        """Preallocate KV cache for efficient inference with position tracking."""
        cache = []
        n_kv_heads = self.cfg.n_heads // 2  # GQA: hardcoded to half the query heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        
        for _ in range(self.cfg.n_layers):
            # Cache in (B, T, H_kv, D) format for efficient indexing
            k_cache = torch.zeros(batch_size, max_seq_len, n_kv_heads, head_dim,
                                 device=device, dtype=dtype)
            v_cache = torch.zeros_like(k_cache)
            pos = 0  # Track current position in cache
            cache.append((k_cache, v_cache, pos))
        return cache

    def forward(self, idx, cache=None):
        x = self.tok(idx)
        
        if cache is None:
            for blk in self.blocks:
                x, _ = blk(x, None)
            return self.head(self.ln_f(x)), None
        
        # With preallocated cache
        new_cache = []
        for i, blk in enumerate(self.blocks):
            x, blk_cache = blk(x, cache[i])
            new_cache.append(blk_cache)

        return self.head(self.ln_f(x)), new_cache