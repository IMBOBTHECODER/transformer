# config.py - Unified configuration for GPU and TPU training
"""
Shared configuration for GPU and TPU training.
TPU-specific settings are automatically applied and cannot be overridden.
"""

from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    """Unified GPT model configuration for GPU and TPU."""
    
    # =========================================================
    # MODEL ARCHITECTURE
    # =========================================================
    vocab_size: int = 32_000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    mlp_ratio: int = 4
    yarn_scale: float = 1.0

    # =========================================================
    # DEVICE-SPECIFIC SETTINGS (will be overridden by device detection)
    # =========================================================
    dtype: torch.dtype = torch.float32
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    use_rope: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # =========================================================
    # MODEL COMPONENTS & ACTIVATION
    # =========================================================
    activation: str = "swiglu"

    # =========================================================
    # OPTIMIZATION HYPERPARAMETERS
    # =========================================================
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    batch_size: int = 128
    sequence_length: int = 512

    # =========================================================
    # REGULARIZATION
    # =========================================================
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    # =========================================================
    # SHARPNESS-AWARE MINIMIZATION (SAM)
    # =========================================================
    sam: bool = False
    sam_rho: float = 0.08

    # =========================================================
    # TRAINING SCHEDULE & TRACKING
    # =========================================================
    init_std: float = 0.02
    id: str = "tinystories"
    total_steps: int = 5_000
    val_interval: int = 200
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_update_interval: float = 0.02
    val_split: float = 0.05

    # =========================================================
    # DATASET CONFIGURATION
    # =========================================================
    dataset_name: str = "Salesforce/wikitext"
    num_samples: int = 350_000
    dataset_split: str = "wikitext-103-raw-v1"  # Can be "train[:250000]" for subset


def apply_tpu_constraints(cfg: GPTConfig) -> GPTConfig:
    """
    Apply TPU-specific constraints to config.
    These settings are MANDATORY for TPU and cannot be overridden.
    """
    cfg.dtype = torch.bfloat16  # TPU: BF16 only
    cfg.compile = False  # TPU: XLA handles compilation
    cfg.gradient_checkpointing = False  # TPU: XLA handles memory
    cfg.use_flash_attention = False  # TPU: SDPA unfused on TPU
    cfg.dropout = 0.0  # TPU: interferes with XLA fusion
    cfg.gradient_clipping = False  # TPU: unnecessary with BF16
    cfg.use_ema = False  # TPU: extra memory transfers, disabled by default
    return cfg


def apply_gpu_device_tuning(cfg: GPTConfig) -> GPTConfig:
    """
    Auto-tune GPU settings based on detected device.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()

            # Modern GPUs: use BF16 and optimizations
            if "a100" in gpu_name or "h100" in gpu_name or "rtx" in gpu_name:
                cfg.dtype = torch.bfloat16
                cfg.use_flash_attention = True
            # Older GPUs: limited to FP32
            else:
                cfg.dtype = torch.float32
                cfg.compile = False
                cfg.gradient_checkpointing = True
                cfg.batch_size = min(cfg.batch_size, 64)
    except Exception:
        pass  # CPU fallback
    
    return cfg
