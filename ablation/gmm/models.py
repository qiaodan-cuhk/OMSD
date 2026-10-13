"""Conditional diagonal Gaussian mixtures used only by the GMM ablation.

Maximum-likelihood pretraining produces an analytic action gradient of log
likelihood. Score normalization defaults to the original ``omiga`` magnitude
scaling; disabling it is an explicit, separate research choice.
The main diffusion workflow never constructs or infers this model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalGMM(nn.Module):
    """A mixture density network for ``p(action | condition)``."""

    def __init__(self, condition_dim: int, action_dim: int, components: int, hidden_dim: int):
        super().__init__()
        for name, value in (("condition_dim", condition_dim), ("action_dim", action_dim),
                            ("components", components), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        self.action_dim = action_dim
        self.components = components
        self.min_std = 1e-4
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, components * (1 + 2 * action_dim)),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.net(condition)
        width = self.components * self.action_dim
        logits = output[:, :self.components]
        means = output[:, self.components:self.components + width].reshape(-1, self.components, self.action_dim)
        log_std = output[:, self.components + width:].reshape(-1, self.components, self.action_dim)
        return logits, means, log_std

    def log_prob(self, action: torch.Tensor,
                 parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Stable per-sample log likelihood with the original softplus scales."""
        logits, means, log_std = parameters
        std = F.softplus(log_std) + self.min_std
        log_normal = -0.5 * (((action[:, None] - means) / std).square()
                             + 2.0 * std.log() + math.log(2.0 * math.pi))
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + log_normal.sum(dim=-1), dim=-1)

    def score(self, action: torch.Tensor, condition: torch.Tensor,
              normalize: bool = True) -> torch.Tensor:
        """Return positive ``∇_action log p``, with frozen GMM weights.

        Normalization preserves the original implementation's magnitude and
        does not change the mathematical score's sign.
        """
        with torch.no_grad():
            parameters = self(condition)
        score = torch.autograd.grad(self.log_prob(action, parameters).sum(), action)[0]
        if normalize:
            score = score / score.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return score.detach()
