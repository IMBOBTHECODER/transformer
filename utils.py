# utils.py
import torch
import torch.nn.functional as F

# =========================================================
# ACTIVATION
# =========================================================

def activation_fn(name: str):
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

    raise ValueError(f"Unary activation '{name}' not supported")



# =========================================================
# PARAMETER GROUPING (DECAY / NO-DECAY)
# =========================================================
def param_groups(model, weight_decay):
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "ln" in name or "bias" in name:
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
                state = self.state.setdefault(p, {})
                m = state.setdefault("m", torch.zeros_like(p))

                m.mul_(b2).add_(p.grad, alpha=1 - b2)
                p.add_(torch.sign(m), alpha=-lr)

                if wd > 0:
                    p.add_(p, alpha=-lr * wd)


# =========================================================
# SAM (SHARPNESS-AWARE MINIMIZATION)
# =========================================================
class SAM:
    def __init__(self, base_optimizer, rho=0.05):
        self.base = base_optimizer
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
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
    def second_step(self):
        self.base.step()
        self.base.zero_grad()


# =========================================================
# OPTIMIZER FACTORY
# =========================================================
def build_optimizer(model, cfg):
    groups = param_groups(model, cfg.weight_decay)
    
    # Check if running on TPU (XLA)
    try:
        import torch_xla.core.xla_model as xm
        has_xla = True
    except ImportError:
        has_xla = False

    if cfg.optimizer == "adamw":
        # TPU: disable fused (not supported on XLA devices)
        # GPU/CPU: use fused for performance
        opt = torch.optim.AdamW(
            groups,
            lr=cfg.lr,
            betas=cfg.betas,
            fused=not has_xla  # Disable fused on TPU
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
        raise ValueError(cfg.optimizer)

    return SAM(opt, cfg.sam_rho) if cfg.sam else opt
