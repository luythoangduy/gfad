from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional

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


class InternalBackboneGate(nn.Module):
    """Trainable gates injected into frozen ViT backbone blocks via forward hooks."""

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        layers: Any = "all",
        mode: str = "multiplication",
        granularity: str = "elementwise",
        activation: str = "sigmoid",
        init_bias: float = 2.0,
    ) -> None:
        super().__init__()
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            raise ValueError("Internal backbone gating currently requires encoder.blocks")
        self.enabled = True
        self.gates = nn.ModuleDict()
        self._handles = []

        resolved_layers = self._resolve_layers(layers, len(blocks))
        for idx in resolved_layers:
            key = str(idx)
            self.gates[key] = BackboneFeatureGate(
                dim=dim,
                mode=mode,
                granularity=granularity,
                activation=activation,
                init_bias=init_bias,
            )
            self._handles.append(blocks[idx].register_forward_hook(self._make_hook(key)))

    @staticmethod
    def _resolve_layers(layers: Any, depth: int) -> list[int]:
        if layers is None:
            layers = "all"
        if isinstance(layers, str):
            name = layers.lower()
            if name == "all":
                return list(range(depth))
            if name == "last":
                return [depth - 1]
            if name.startswith("last_"):
                count = int(name.split("_", 1)[1])
                return list(range(max(depth - count, 0), depth))
            layers = [int(name)]
        elif isinstance(layers, int):
            layers = [layers]
        elif not isinstance(layers, Iterable):
            raise ValueError("backbone_gating.layers must be 'all', 'last', 'last_N', an int, or a list of ints")

        resolved = []
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx = depth + idx
            if idx < 0 or idx >= depth:
                raise ValueError(f"backbone_gating layer {layer} is outside backbone depth {depth}")
            resolved.append(idx)
        return sorted(set(resolved))

    def _make_hook(self, key: str):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return output
            gate = self.gates[key]
            if isinstance(output, tuple):
                return (gate(output[0]), *output[1:])
            if isinstance(output, list):
                return [gate(output[0]), *output[1:]]
            return gate(output)

        return hook

    @contextmanager
    def use_gates(self, enabled: bool):
        prev = self.enabled
        self.enabled = enabled
        try:
            yield
        finally:
            self.enabled = prev

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def build_backbone_gate(
    encoder: nn.Module,
    dim: int,
    config: Optional[Mapping[str, Any]],
) -> Optional[InternalBackboneGate]:
    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return None
    return InternalBackboneGate(
        encoder=encoder,
        dim=dim,
        layers=cfg.get("layers", "all"),
        mode=str(cfg.get("mode", "multiplication")).lower(),
        granularity=str(cfg.get("granularity", "elementwise")).lower(),
        activation=str(cfg.get("activation", "sigmoid")).lower(),
        init_bias=float(cfg.get("init_bias", 2.0)),
    )
