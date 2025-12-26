# utils.py (TPU)

from typing import List, Dict, Callable, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import time

# TPU imports
try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# =========================================================
# ACTIVATION FUNCTIONS
# =========================================================

def activation_fn(name: str) -> Callable:
    """
    Get activation function by name.
    
    Args:
        name: relu, relu2, gelu, silu, mish, or snake
    
    Returns:
        Callable activation function
    """
    name = name.lower()

    if name == "relu":
        return F.relu
    if name == "relu2":
        return lambda x: F.relu(x) ** 2
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    if name == "mish":
        return F.mish
    if name == "snake":
        return lambda x: x + torch.sin(x) ** 2

    raise ValueError(f"Unknown activation: {name}")



# =========================================================
# PARAMETER GROUPING
# =========================================================

def param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """
    Separate parameters into decay and no-decay groups.
    
    No decay: biases, layer norms, embeddings
    Decay: all other weights (matrices)
    
    Args:
        model: PyTorch model
        weight_decay: L2 regularization strength
    
    Returns:
        List of parameter groups for optimizer
    """
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "ln" in name or "norm" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# =========================================================
# LION OPTIMIZER
# =========================================================

class Lion(torch.optim.Optimizer):
    """
    Lion (EvoLved sIgn Momentum) optimizer.
    
    Advantages: faster convergence, lower memory, better generalization via sign-based updates.
    Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
    """
    
    def __init__(self, params, lr: float, betas: Tuple[float, float] = (0.9, 0.99), wd: float = 0.0):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            betas: (beta1, beta2) momentum coefficients
            wd: Weight decay
        """
        super().__init__(params, dict(lr=lr, betas=betas, wd=wd))

    @torch.no_grad()
    def step(self) -> None:
        """Optimizer step with sign-based momentum."""
        for g in self.param_groups:
            lr, wd = g["lr"], g["wd"]
            _, b2 = g["betas"]

            for p in g["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                m = state.setdefault("m", torch.zeros_like(p))

                # Momentum update
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
                
                # Sign-based update (key difference from Adam)
                p.add_(torch.sign(m), alpha=-lr)

                # Decoupled weight decay
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)


# =========================================================
# SAM: SHARPNESS-AWARE MINIMIZATION
# =========================================================

class SAM:
    """
    Sharpness-Aware Minimization for flatter minima.
    
    Two forward passes per step: perturb weights, compute loss at perturbed location, then update.
    Better generalization via flatter loss landscape minima.
    
    Reference: Foret et al., "Sharpness-Aware Minimization" (ICLR 2021)
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        """
        Args:
            base_optimizer: Underlying optimizer (AdamW, Lion, etc.)
            rho: Perturbation radius (typically 0.01-0.1)
        """
        self.base = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self) -> None:
        """Perturb weights toward gradient direction."""
        norm = torch.norm(torch.stack([
            p.grad.norm()
            for g in self.base.param_groups
            for p in g["params"]
            if p.grad is not None
        ]))
        scale = self.rho / (norm + 1e-12)

        for g in self.base.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self) -> None:
        """Update weights at perturbed location."""
        self.base.step()
        self.base.zero_grad()


# =========================================================
# OPTIMIZER BUILDER
# =========================================================

def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """
    Build optimizer for TPU training.
    
    Disables fused kernels (not supported on XLA), uses decoupled weight decay.
    Can optionally wrap with SAM for better generalization.
    
    Args:
        model: PyTorch model to optimize
        cfg: Config with optimizer settings (optimizer, lr, betas, weight_decay, sam, sam_rho)
    
    Returns:
        Optimizer instance (possibly SAM-wrapped)
    """
    groups = param_groups(model, cfg.weight_decay)

    if cfg.optimizer == "adamw":
        # AdamW with fused=False for XLA (fused kernels not supported)
        opt = torch.optim.AdamW(
            groups,
            lr=cfg.lr,
            betas=cfg.betas,
            fused=False
        )
    elif cfg.optimizer == "lion":
        # Lion: naturally XLA-compatible
        opt = Lion(groups, lr=cfg.lr, wd=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        # SGD with Nesterov momentum
        opt = torch.optim.SGD(
            groups,
            lr=cfg.lr,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Optionally wrap with SAM
    if hasattr(cfg, 'sam') and cfg.sam:
        return SAM(opt, cfg.sam_rho)
    return opt


# =========================================================
# TRAINING STATISTICS & MONITORING
# =========================================================

@dataclass
class TrainingStats:
    """Track training statistics for logging and analysis."""
    
    step: int = 0
    loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    tokens_processed: int = 0
    throughput_toks_per_sec: float = 0.0
    elapsed_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format for logging."""
        return (
            f"Step {self.step:5d} | "
            f"Loss {self.loss:.4f} | "
            f"Val Loss {self.val_loss:.4f} | "
            f"LR {self.learning_rate:.2e} | "
            f"Throughput {self.throughput_toks_per_sec:.0f} tok/s"
        )


class TPUProfiler:
    """Profile and monitor TPU execution and memory usage."""
    
    def __init__(self, enabled: bool = False):
        """
        Args:
            enabled: Enable profiling (has overhead)
        """
        self.enabled = enabled and HAS_XLA
        self.profile_data = []
    
    def start_step(self) -> float:
        """Start timing a training step."""
        return time.perf_counter()
    
    def end_step(self, start_time: float, step: int) -> float:
        """Record step duration and return elapsed time."""
        elapsed = time.perf_counter() - start_time
        if self.enabled:
            self.profile_data.append({"step": step, "elapsed": elapsed})
        return elapsed
    
    def get_xla_metrics(self) -> Optional[str]:
        """Get XLA compilation and execution metrics."""
        if not self.enabled or not HAS_XLA:
            return None
        try:
            return xm.metrics_report()
        except Exception as e:
            return f"Failed to get metrics: {e}"
    
    def summary(self) -> Optional[str]:
        """Print profiling summary."""
        if not self.profile_data or not self.enabled:
            return None
        
        times = [p["elapsed"] for p in self.profile_data]
        avg_time = sum(times) / len(times) if times else 0
        return (
            f"\nProfiling Summary:\n"
            f"  Steps profiled: {len(times)}\n"
            f"  Avg step time: {avg_time*1000:.2f}ms\n"
            f"  Min step time: {min(times)*1000:.2f}ms\n"
            f"  Max step time: {max(times)*1000:.2f}ms"
        )
    
    def reset(self) -> None:
        """Clear profiling data."""
        self.profile_data = []


class MetricsBuffer:
    """Buffer metrics to reduce device syncs."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.losses = []
        self.learning_rates = []
        self.timestamps = []
    
    def add(self, loss: torch.Tensor, lr: float) -> None:
        """Add metric without syncing."""
        self.losses.append(loss.detach())
        self.learning_rates.append(lr)
        self.timestamps.append(time.perf_counter())
    
    def reduce(self) -> Tuple[float, float]:
        """
        Reduce buffered metrics (single device sync).
        
        Returns:
            (mean_loss, mean_lr)
        """
        if not self.losses:
            return 0.0, 0.0
        
        # Stack all losses at once and sync once
        loss_tensor = torch.stack(self.losses).mean()
        mean_lr = sum(self.learning_rates) / len(self.learning_rates)
        
        # Single sync for all metrics
        if HAS_XLA:
            xm.mark_step()
        
        mean_loss = loss_tensor.item() if isinstance(loss_tensor, torch.Tensor) else float(loss_tensor)
        
        self.clear()
        return mean_loss, mean_lr
    
    def clear(self) -> None:
        """Clear buffers."""
        self.losses.clear()
        self.learning_rates.clear()
        self.timestamps.clear()
    
    def should_flush(self) -> bool:
        """Check if buffer should be reduced."""
        return len(self.losses) >= self.max_size


def get_device_info() -> Dict[str, Any]:
    """Get information about current TPU device."""
    info = {"has_tpu": HAS_XLA, "has_cuda": torch.cuda.is_available()}
    
    if HAS_XLA:
        try:
            info["tpu_device"] = str(xm.xla_device())
            info["tpu_device_count"] = xm.get_world_size()
            info["tpu_rank"] = xm.get_ordinal()
        except Exception as e:
            info["tpu_error"] = str(e)
    
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    
    return info


def print_device_info() -> None:
    """Print device configuration."""
    info = get_device_info()
    
    print("\n" + "="*50)
    print("DEVICE CONFIGURATION")
    print("="*50)
    
    if info["has_tpu"]:
        print(f"✓ TPU Detected: {info.get('tpu_device', 'N/A')}")
        print(f"  Rank: {info.get('tpu_rank', '?')}/{info.get('tpu_device_count', '?')}")
    elif info["has_cuda"]:
        print(f"✓ GPU Detected: {info.get('cuda_device', 'N/A')}")
    else:
        print("ℹ Using CPU")
    
    print("="*50 + "\n")
