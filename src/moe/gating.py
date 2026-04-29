"""
Gating networks for Mixture-of-Experts layers.

Provides two variants:
  - TopKGating: tokens choose their top-k experts.
  - ExpertChoiceGating: experts choose their top-k tokens.

Both optionally compute an auxiliary load-balancing loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """
    Standard top-k gating: each token selects its *top_k* experts.

    Forward signature::

        gate_values, expert_indices, aux_loss = gating(hidden_states)

    Shapes
    ------
    - hidden_states: ``(batch, seq_len, hidden_size)``
    - gate_values:   ``(batch, seq_len, top_k)``  — softmax weights for chosen experts
    - expert_indices: ``(batch, seq_len, top_k)`` — expert ids (long)
    - aux_loss:       scalar — load-balancing auxiliary loss (0 when disabled)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        load_balance_weight: float = 0.01,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: (B, S, H)
        logits = self.gate(hidden_states)  # (B, S, E)

        # Optional noise for exploration during training
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Softmax over experts
        probs = F.softmax(logits, dim=-1)  # (B, S, E)

        # Select top-k experts per token
        gate_values, expert_indices = torch.topk(probs, self.top_k, dim=-1)

        # Re-normalise so selected gate values sum to 1
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-9)

        # Auxiliary load-balancing loss (Switch Transformer style)
        aux_loss = self._load_balance_loss(probs, expert_indices)

        return gate_values, expert_indices, aux_loss

    def _load_balance_loss(
        self, probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary load-balancing loss.

        L_balance = E * sum_i(f_i * P_i)

        where f_i is the fraction of tokens dispatched to expert i
        and P_i is the average routing probability for expert i.
        """
        if self.load_balance_weight == 0.0:
            return torch.tensor(0.0, device=probs.device)

        # Flatten batch and sequence dims
        B, S, _ = probs.shape
        flat_probs = probs.view(-1, self.num_experts)         # (B*S, E)
        flat_indices = expert_indices.view(-1, self.top_k)     # (B*S, top_k)

        # f_i: fraction of tokens routed to each expert
        counts = torch.zeros(self.num_experts, device=probs.device)
        for k in range(self.top_k):
            counts.scatter_add_(
                0, flat_indices[:, k], torch.ones(B * S, device=probs.device)
            )
        f = counts / (B * S * self.top_k + 1e-9)

        # P_i: mean routing probability per expert
        P = flat_probs.mean(dim=0)

        loss = self.load_balance_weight * self.num_experts * (f * P).sum()
        return loss


class ExpertChoiceGating(nn.Module):
    """
    Expert-choice gating: each expert selects its top-C tokens.

    This variant guarantees perfectly balanced expert utilisation at the cost
    of some tokens potentially being processed by zero experts.

    Forward signature::

        dispatch_mask, combine_weights, aux_loss = gating(hidden_states)

    Shapes
    ------
    - hidden_states:    ``(batch * seq_len, hidden_size)`` — flattened token dim
    - dispatch_mask:    ``(num_experts, capacity)`` — token indices chosen by each expert (long)
    - combine_weights:  ``(num_experts, capacity)`` — softmax weights for chosen tokens
    - aux_loss:         scalar (always 0 for this variant, included for API symmetry)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.top_k = top_k

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (N, H) where N = total tokens = B * S.
        """
        N, H = hidden_states.shape
        capacity = int(N * self.top_k * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)

        logits = self.gate(hidden_states)          # (N, E)
        probs = F.softmax(logits, dim=0)           # softmax over tokens (column-wise)

        # Each expert picks its top-C tokens
        # Transpose so experts are rows: (E, N)
        probs_t = probs.t()
        combine_weights, dispatch_mask = torch.topk(probs_t, capacity, dim=-1)
        # dispatch_mask: (E, C) — token indices
        # combine_weights: (E, C) — weights

        # Re-normalise weights per expert
        combine_weights = combine_weights / (combine_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # No aux loss needed — expert utilisation is balanced by construction
        aux_loss = torch.tensor(0.0, device=hidden_states.device)

        return dispatch_mask, combine_weights, aux_loss
