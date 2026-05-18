from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SUPPORTED_GATE_MODES = {"multiplication", "residual"}
SUPPORTED_GATE_POSITIONS = ("pre_qkv", "q", "k", "v", "attn_output", "proj_output")
POSITION_ALIASES = {
    "input": "pre_qkv",
    "query": "q",
    "key": "k",
    "value": "v",
    "post_attention": "attn_output",
    "post_attn": "attn_output",
    "pre_proj": "attn_output",
    "projection": "proj_output",
    "post_proj": "proj_output",
    "output": "proj_output",
}


@dataclass(frozen=True)
class GateSpec:
    position: str
    mode: str
    granularity: str


def _as_dict(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(config or {})


def _canonical_position(position: str) -> str:
    position = str(position).lower()
    return POSITION_ALIASES.get(position, position)


def _normalize_gate_specs(config: Mapping[str, Any]) -> List[GateSpec]:
    default_mode = str(config.get("mode", config.get("integration_mode", "multiplication"))).lower()
    default_granularity = str(config.get("granularity", "elementwise")).lower()
    raw_positions = config.get("positions", ["attn_output"])

    if isinstance(raw_positions, str):
        raw_positions = [raw_positions]
    raw_positions = list(raw_positions)
    if len(raw_positions) == 1 and str(raw_positions[0]).lower() == "all":
        raw_positions = list(SUPPORTED_GATE_POSITIONS)

    specs: List[GateSpec] = []
    for entry in raw_positions:
        if isinstance(entry, Mapping):
            position = _canonical_position(entry.get("position", entry.get("name", "attn_output")))
            mode = str(entry.get("mode", default_mode)).lower()
            granularity = str(entry.get("granularity", default_granularity)).lower()
        else:
            position = _canonical_position(entry)
            mode = default_mode
            granularity = default_granularity

        if position not in SUPPORTED_GATE_POSITIONS:
            raise ValueError(
                f"Unsupported gated attention position '{position}'. "
                f"Supported positions: {sorted(SUPPORTED_GATE_POSITIONS)}"
            )
        if mode not in SUPPORTED_GATE_MODES:
            raise ValueError(
                f"Unsupported gated attention mode '{mode}'. "
                f"Supported modes: {sorted(SUPPORTED_GATE_MODES)}"
            )
        if granularity not in {"elementwise", "headwise", "token"}:
            raise ValueError("gated_attention.granularity must be one of: elementwise, headwise, token")

        specs.append(GateSpec(position=position, mode=mode, granularity=granularity))

    return specs


class GateUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str,
        granularity: str,
        activation: str = "sigmoid",
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if granularity not in {"elementwise", "headwise", "token"}:
            raise ValueError(f"Unsupported granularity: {granularity}")
        self.mode = mode
        self.granularity = granularity
        self.activation = activation
        out_dim = dim if granularity == "elementwise" else 1
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def _activate(self, gate: Tensor) -> Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(gate)
        if self.activation == "tanh":
            return torch.tanh(gate)
        if self.activation == "silu":
            return torch.nn.functional.silu(gate)
        if self.activation == "identity":
            return gate
        raise ValueError(f"Unsupported gated_attention.activation: {self.activation}")

    def forward(self, x: Tensor) -> Tensor:
        gate = self._activate(self.proj(x))
        self.last_gate = gate
        if self.mode == "multiplication":
            return x * gate
        if self.mode == "residual":
            return x + (x * gate)
        raise ValueError(f"Unsupported gated attention mode: {self.mode}")


class GatedSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gate_specs: Optional[Iterable[GateSpec]] = None,
        activation: str = "sigmoid",
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gates = nn.ModuleDict()
        for spec in gate_specs or []:
            gate_dim = dim if spec.position in {"pre_qkv", "proj_output"} else self.head_dim
            self.gates[spec.position] = GateUnit(
                dim=gate_dim,
                mode=spec.mode,
                granularity=spec.granularity,
                activation=activation,
                init_bias=init_bias,
            )

    @classmethod
    def from_attention(cls, attention: nn.Module, config: Mapping[str, Any]) -> "GatedSelfAttention":
        specs = _normalize_gate_specs(config)
        gated = cls(
            dim=attention.qkv.in_features,
            num_heads=attention.num_heads,
            qkv_bias=attention.qkv.bias is not None,
            proj_bias=attention.proj.bias is not None,
            attn_drop=attention.attn_drop.p,
            proj_drop=attention.proj_drop.p,
            gate_specs=specs,
            activation=str(config.get("activation", "sigmoid")).lower(),
            init_bias=float(config.get("init_bias", 0.0)),
        )
        gated.qkv.load_state_dict(attention.qkv.state_dict())
        gated.proj.load_state_dict(attention.proj.state_dict())
        return gated

    def _gate(self, position: str, x: Tensor) -> Tensor:
        gate = self.gates[position] if position in self.gates else None
        return gate(x) if gate is not None else x

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if attn_bias is not None:
            raise AssertionError("GatedSelfAttention does not support nested tensor attention bias")

        B, N, C = x.shape
        x = self._gate("pre_qkv", x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = self._gate("q", qkv[0]) * self.scale
        k = self._gate("k", qkv[1])
        v = self._gate("v", qkv[2])

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self._gate("attn_output", x)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self._gate("proj_output", x)
        x = self.proj_drop(x)
        return x


def apply_gated_attention_to_predictor(predictor: nn.Module, config: Optional[Mapping[str, Any]]) -> nn.Module:
    cfg = _as_dict(config)
    if not cfg.get("enabled", False):
        return predictor

    for block in predictor.predictor_blocks:
        block.attn = GatedSelfAttention.from_attention(block.attn, cfg)
    predictor.gated_attention_config = cfg
    return predictor


def collect_gate_keep_maps(predictor: nn.Module, positions: Optional[Iterable[str]] = None) -> Optional[Tensor]:
    if isinstance(positions, str):
        positions = [positions]
    if positions is not None:
        positions = {_canonical_position(position) for position in positions}
        if "all" in positions:
            positions = None

    gate_maps: List[Tensor] = []
    for block in getattr(predictor, "predictor_blocks", []):
        attn = getattr(block, "attn", None)
        gates = getattr(attn, "gates", {})
        for position, gate in gates.items():
            if positions is not None and position not in positions:
                continue
            gate_value = getattr(gate, "last_gate", None)
            if gate_value is None:
                continue
            if gate.activation != "sigmoid":
                gate_value = torch.sigmoid(gate_value)
            if gate_value.ndim == 4:
                gate_value = gate_value.mean(dim=(1, 3))
            elif gate_value.ndim == 3:
                gate_value = gate_value.mean(dim=-1)
            else:
                continue
            gate_maps.append(gate_value)

    if not gate_maps:
        return None
    return torch.stack(gate_maps, dim=0).mean(dim=0)


def gate_supervision_loss(
    predictor: nn.Module,
    anomaly_mask: Tensor,
    num_tokens: int,
    positions: Optional[Iterable[str]] = None,
    anomaly_weight: float = 4.0,
) -> Optional[Tensor]:
    gate_keep = collect_gate_keep_maps(predictor, positions=positions)
    if gate_keep is None:
        return None

    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(f"Cannot reshape {num_tokens} tokens into a square anomaly map")

    gate_keep = gate_keep.float()
    anomaly_tokens = F.interpolate(anomaly_mask.float(), size=(side, side), mode="area").flatten(1)
    anomaly_tokens = (anomaly_tokens > 0).to(gate_keep.dtype)
    keep_target = 1.0 - anomaly_tokens
    weights = 1.0 + anomaly_tokens * float(anomaly_weight)
    gate_keep = gate_keep.clamp(min=1e-4, max=1.0 - 1e-4)
    return F.binary_cross_entropy(gate_keep, keep_target, weight=weights, reduction="mean")
