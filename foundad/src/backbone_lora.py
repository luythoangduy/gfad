from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


class LoRAResidualAdapter(nn.Module):
    """Low-rank residual adapter applied to frozen backbone block outputs."""

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("backbone_lora.rank must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.Linear(dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.dropout(self.down(x))) * self.scale


class InternalBackboneLoRA(nn.Module):
    """Inject LoRA-style adapters into frozen ViT backbone blocks via hooks."""

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        layers: Any = "all",
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            raise ValueError("Internal backbone LoRA currently requires encoder.blocks")
        self.enabled = True
        self.adapters = nn.ModuleDict()
        self._handles = []

        resolved_layers = self._resolve_layers(layers, len(blocks))
        for idx in resolved_layers:
            key = str(idx)
            self.adapters[key] = LoRAResidualAdapter(
                dim=dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
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
            raise ValueError("backbone_lora.layers must be 'all', 'last', 'last_N', an int, or a list of ints")

        resolved = []
        for layer in layers:
            idx = int(layer)
            if idx < 0:
                idx = depth + idx
            if idx < 0 or idx >= depth:
                raise ValueError(f"backbone_lora layer {layer} is outside backbone depth {depth}")
            resolved.append(idx)
        return sorted(set(resolved))

    def _make_hook(self, key: str):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return output
            adapter = self.adapters[key]
            if isinstance(output, tuple):
                return (adapter(output[0]), *output[1:])
            if isinstance(output, list):
                return [adapter(output[0]), *output[1:]]
            return adapter(output)

        return hook

    @contextmanager
    def use_adapters(self, enabled: bool):
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


def build_backbone_lora(
    encoder: nn.Module,
    dim: int,
    config: Optional[Mapping[str, Any]],
) -> Optional[InternalBackboneLoRA]:
    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return None
    return InternalBackboneLoRA(
        encoder=encoder,
        dim=dim,
        layers=cfg.get("layers", "all"),
        rank=int(cfg.get("rank", 16)),
        alpha=float(cfg.get("alpha", cfg.get("rank", 16))),
        dropout=float(cfg.get("dropout", 0.0)),
    )
