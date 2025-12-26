# utils.py (GPU-optimized)
"""GPU-specific optimizers and utilities for efficient training."""

import torch
import torch.nn.functional as F

# =========================================================
# ACTIVATION FUNCTIONS
# =========================================================
def activation_fn(name: str):
    """Get activation function by name."""
    name = name.lower()
    
    activations = {
        "relu": F.relu,
        "relu2": lambda x: F.relu(x) ** 2,
        "gelu": F.gelu,
        "silu": F.silu,
        "mish": F.mish,
        "snake": lambda x: x + torch.sin(x) ** 2,
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]



# =========================================================
# PARAMETER GROUPING (Weight Decay / No Decay)
# =========================================================
def param_groups(model, weight_decay):
    """Separate parameters into decay and no-decay groups for AdamW."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No decay: low-dim params (bias, norm weights)
        if p.dim() < 2 or "ln" in name or "bias" in name:
            no_decay.append(p)
        # Decay: high-dim params (linear layers)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# =========================================================
# LION OPTIMIZER (Sign-Based Update)
# =========================================================
class Lion(torch.optim.Optimizer):
    """Lion: weight decay generalization to all parameters. Uses sign(m) instead of m.
    
    Reference: https://arxiv.org/abs/2302.06675
    Often outperforms AdamW with larger learning rates and less warmup.
    """
    
    def __init__(self, params, lr, betas=(0.9, 0.99), wd=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, wd=wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, wd = g["lr"], g["wd"]
            _, b2 = g["betas"]

            for p in g["params"]:
                if p.grad is None:
                    continue
                
                # Maintain exponential moving average of gradients
                state = self.state.setdefault(p, {})
                m = state.setdefault("m", torch.zeros_like(p))
                
                # Update bias: m_t = β₂ * m_{t-1} + (1 - β₂) * g_t
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
                
                # Update params: θ_t = θ_{t-1} - lr * sign(m_t)
                p.add_(torch.sign(m), alpha=-lr)
                
                # Weight decay: θ_t = θ_t - lr * λ * θ_{t-1}
                if wd > 0:
                    p.add_(p, alpha=-lr * wd)


# =========================================================
# SAM (Sharpness-Aware Minimization)
# =========================================================
class SAM:
    """Sharpness-Aware Minimization: seeks flat minima for better generalization.
    
    Two-step process: perturb weights in gradient direction, compute loss on 
    perturbed weights, then step with that gradient. Adds ~100% compute overhead.
    """
    
    def __init__(self, base_optimizer, rho=0.05):
        self.base = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        """Perturb weights: θ → θ + ρ/‖∇L‖ * ∇L"""
        # Compute gradient norm across all parameters
        grad_norm = torch.norm(torch.stack([
            p.grad.norm() for g in self.base.param_groups
            for p in g["params"] if p.grad is not None
        ]))
        
        # Scale factor for perturbation
        scale = self.rho / (grad_norm + 1e-12)
        
        # Apply perturbation: θ += scale * ∇L
        for g in self.base.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        """Step with loss gradient at perturbed weights."""
        self.base.step()
        self.base.zero_grad()


# =========================================================
# OPTIMIZER FACTORY
# =========================================================
def build_optimizer(model, cfg):
    """Build optimizer with parameter grouping and optional SAM wrapper."""
    groups = param_groups(model, cfg.weight_decay)
    
    if cfg.optimizer == "adamw":
        # GPU: use fused AdamW for faster kernels
        opt = torch.optim.AdamW(
            groups,
            lr=cfg.lr,
            betas=cfg.betas,
            fused=True  # CUDA-optimized kernel
        )
    elif cfg.optimizer == "lion":
        opt = Lion(groups, lr=cfg.lr, wd=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            groups,
            lr=cfg.lr,
            momentum=0.9,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # Wrap with SAM if enabled (2x compute, better generalization)
    return SAM(opt, cfg.sam_rho) if cfg.sam else opt
