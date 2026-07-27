"""
Muon Optimizer (Momentum Orthogonalized by Newton-Schulz)
Pre-training optimizer designed for 2D matrix weights in Deep Transformer models.
Reference: Keller Jordan / modded-nanogpt (2024-2026 SOTA LLM optimization).
"""

import torch

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the matrix square-root inverse / orthogonalization of G.
    """
    assert G.ndim == 2, f"Muon zeropower requires 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        X = X.T
    else:
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz
    Optimizes 2D matrix parameters via orthogonalized momentum updates.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, n_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, n_steps=n_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            n_steps = group['n_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if p.ndim == 2:
                    update = zeropower_via_newtonschulz5(buf, steps=n_steps)
                    scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                    p.data.add_(update, alpha=-lr * scale)
                else:
                    p.data.add_(buf, alpha=-lr)

        return loss
