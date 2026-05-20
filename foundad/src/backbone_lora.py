from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


class LoRAWeights(nn.Module):
    """Trainable low-rank delta for a frozen Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
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
        self.enabled = True
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.dropout(self.down(x))) * self.scale


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable LoRA delta."""

    def __init__(self, base: nn.Linear, adapter: LoRAWeights) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.base.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if not self.adapter.enabled:
            return y
        return y + self.adapter(x)


class InternalBackboneLoRA(nn.Module):
    """Inject LoRA into Linear layers inside each selected ViT block MLP."""

    def __init__(
        self,
        encoder: nn.Module,
        layers: Any = "all",
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Any = None,
    ) -> None:
        super().__init__()
        blocks = getattr(encoder, "blocks", None)
        if blocks is None:
            raise ValueError("Internal backbone LoRA currently requires encoder.blocks")

        self.adapters = nn.ModuleDict()
        self._wrapped: list[tuple[int, str]] = []
        self._target_modules = self._normalize_target_modules(target_modules)

        resolved_layers = self._resolve_layers(layers, len(blocks))
        for idx in resolved_layers:
            mlp = getattr(blocks[idx], "mlp", None)
            if mlp is None:
                raise ValueError(f"encoder.blocks[{idx}] does not expose an mlp module")
            for name, linear in self._iter_target_linears(mlp):
                adapter_key = self._adapter_key(idx, name)
                adapter = LoRAWeights(
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                self.adapters[adapter_key] = adapter
                self._replace_submodule(mlp, name, LoRALinear(linear, adapter))
                self._wrapped.append((idx, name))

        if not self._wrapped:
            targets = ", ".join(sorted(self._target_modules)) if self._target_modules else "all Linear modules"
            raise ValueError(f"No MLP Linear layers matched backbone_lora.target_modules={targets}")

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

    @staticmethod
    def _normalize_target_modules(target_modules: Any) -> Optional[set[str]]:
        if target_modules is None:
            return None
        if isinstance(target_modules, str):
            if target_modules.lower() == "all":
                return None
            target_modules = [target_modules]
        return {str(name) for name in target_modules}

    @staticmethod
    def _adapter_key(layer_idx: int, module_name: str) -> str:
        safe_name = module_name.replace(".", "__")
        return f"block_{layer_idx}__mlp__{safe_name}"

    def _iter_target_linears(self, mlp: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
        for name, module in mlp.named_modules():
            if not name or isinstance(module, LoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                continue
            leaf_name = name.rsplit(".", 1)[-1]
            if self._target_modules is not None and name not in self._target_modules and leaf_name not in self._target_modules:
                continue
            yield name, module

    @staticmethod
    def _replace_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
        parent_name, _, child_name = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, new_module)

    @contextmanager
    def use_adapters(self, enabled: bool):
        previous = [adapter.enabled for adapter in self.adapters.values()]
        for adapter in self.adapters.values():
            adapter.enabled = enabled
        try:
            yield
        finally:
            for adapter, was_enabled in zip(self.adapters.values(), previous):
                adapter.enabled = was_enabled


def build_backbone_lora(
    encoder: nn.Module,
    dim: int,
    config: Optional[Mapping[str, Any]],
) -> Optional[InternalBackboneLoRA]:
    del dim
    cfg = dict(config or {})
    if not cfg.get("enabled", False):
        return None
    return InternalBackboneLoRA(
        encoder=encoder,
        layers=cfg.get("layers", "all"),
        rank=int(cfg.get("rank", 16)),
        alpha=float(cfg.get("alpha", cfg.get("rank", 16))),
        dropout=float(cfg.get("dropout", 0.0)),
        target_modules=cfg.get("target_modules"),
    )
