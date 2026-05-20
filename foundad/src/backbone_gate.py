from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn


class BackboneFeatureGate(nn.Module):
    """A lightweight trainable gate on frozen backbone patch features."""

    def __init__(
        self,
        dim: int,
        mode: str = "multiplication",
        granularity: str = "elementwise",
        activation: str = "sigmoid",
        init_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in {"multiplication", "residual"}:
            raise ValueError("backbone_gating.mode must be multiplication or residual")
        if granularity not in {"elementwise", "token"}:
            raise ValueError("backbone_gating.granularity must be elementwise or token")

        self.mode = mode
        self.granularity = granularity
        self.activation = activation
        out_dim = dim if granularity == "elementwise" else 1
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)
        self.last_gate: Optional[Tensor] = None

    def _activate(self, x: Tensor) -> Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "silu":
            return torch.nn.functional.silu(x)
        if self.activation == "identity":
            return x
        raise ValueError(f"Unsupported backbone_gating.activation: {self.activation}")

    def forward(self, x: Tensor) -> Tensor:
        gate = self._activate(self.proj(x))
        self.last_gate = gate
        if self.mode == "multiplication":
            return x * gate
        if self.mode == "residual":
            return x + x * gate
        raise ValueError(f"Unsupported backbone_gating.mode: {self.mode}")


def build_backbone_gate(dim: int, config: Optional[Mapping[str, Any]]) -> Optional[BackboneFeatureGate]:
    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return None
    return BackboneFeatureGate(
        dim=dim,
        mode=str(cfg.get("mode", "multiplication")).lower(),
        granularity=str(cfg.get("granularity", "elementwise")).lower(),
        activation=str(cfg.get("activation", "sigmoid")).lower(),
        init_bias=float(cfg.get("init_bias", 2.0)),
    )
