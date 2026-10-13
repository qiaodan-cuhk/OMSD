"""Explicit GMM training implementation, isolated from diffusion entrypoints."""

from __future__ import annotations

import torch
from torch import nn

from omsd.agents import OfflineAgent

from .models import ConditionalGMM


class GMMOfflineAgent(OfflineAgent):
    """Use the same MARL decomposition and IQL, with GMM behavior equations.

    The dedicated pretraining and policy scripts construct this class directly;
    checkpoint contents never select the implementation.
    """

    behavior_model = "gmm"

    def _validate_behavior_config(self) -> None:
        for name in ("n_gmm_components", "gmm_hidden_dim"):
            value = getattr(self.config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")

    def _make_behavior(self, condition_dim: int, action_dim: int) -> nn.Module:
        return ConditionalGMM(condition_dim, action_dim, self.config.n_gmm_components,
                              self.config.gmm_hidden_dim)

    def _behavior_loss(self, behavior: ConditionalGMM, action: torch.Tensor,
                       condition: torch.Tensor) -> torch.Tensor:
        return -behavior.log_prob(action, behavior(condition)).mean()

    def _behavior_vector(self, behavior: ConditionalGMM, action: torch.Tensor,
                         condition: torch.Tensor) -> torch.Tensor:
        # The shared loss is minimized, so its regularizer uses NEGATIVE score.
        # Gradient descent then follows +∇log p, as in the paper's Eq. (5)-(6).
        # Default normalization keeps the original GMM regularizer magnitude.
        return -behavior.score(action.detach().requires_grad_(True), condition,
                               normalize=self.config.gmm_score_normalize)
