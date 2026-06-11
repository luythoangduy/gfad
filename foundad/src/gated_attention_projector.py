from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import Tensor, nn


SUPPORTED_GATE_MODES = {"multiplication", "residual", "memory"}
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
        if mode == "memory":
            self.keep_proj = nn.Linear(dim, out_dim)
            self.write_proj = nn.Linear(dim, out_dim)
            self.candidate_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.keep_proj.weight)
            nn.init.constant_(self.keep_proj.bias, init_bias)
            nn.init.zeros_(self.write_proj.weight)
            nn.init.constant_(self.write_proj.bias, init_bias)
        else:
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

    def forward(self, x: Tensor, memory: Optional[Tensor] = None) -> Tensor:
        if self.mode == "memory":
            memory = x if memory is None else memory
            keep = torch.sigmoid(self.keep_proj(memory))
            write = torch.sigmoid(self.write_proj(memory))
            candidate = torch.tanh(self.candidate_proj(x))
            return keep * memory + write * candidate

        gate = self._activate(self.proj(x))
        if self.mode == "multiplication":
            return x * gate
        if self.mode == "residual":
            return x + (x * gate)
        raise ValueError(f"Unsupported gated attention mode: {self.mode}")


SUPPORTED_MEMORY_TYPES = {"self", "patch_mean"}


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
        memory_type: str = "self",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if memory_type not in SUPPORTED_MEMORY_TYPES:
            raise ValueError(
                f"Unsupported memory_type '{memory_type}'. "
                f"Supported: {sorted(SUPPORTED_MEMORY_TYPES)}"
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.memory_type = memory_type

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
            memory_type=str(config.get("memory_type", "self")).lower(),
        )
        gated.qkv.load_state_dict(attention.qkv.state_dict())
        gated.proj.load_state_dict(attention.proj.state_dict())
        return gated

    def _gate(self, position: str, x: Tensor, memory: Optional[Tensor] = None) -> Tensor:
        gate = self.gates[position] if position in self.gates else None
        return gate(x, memory=memory) if gate is not None else x

    def _build_memory(self, x: Tensor) -> Tensor:
        """Compute the memory tensor used to condition the gate.

        memory_type='self'       : each patch uses itself as reference (original behaviour).
        memory_type='patch_mean' : all patches share the image-level mean as reference,
                                   giving the gate a stable, anomaly-agnostic context signal.
        """
        if self.memory_type == "patch_mean":
            # mean over patch dimension → [B, 1, C] → expand to [B, N, C]
            return x.mean(dim=1, keepdim=True).expand_as(x)
        # default: 'self'
        return x

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if attn_bias is not None:
            raise AssertionError("GatedSelfAttention does not support nested tensor attention bias")

        B, N, C = x.shape
        memory = self._build_memory(x)   # [B, N, C]  — 'self' or 'patch_mean'
        memory_heads = memory.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self._gate("pre_qkv", x, memory=memory)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = self._gate("q", qkv[0], memory=memory_heads) * self.scale
        k = self._gate("k", qkv[1], memory=memory_heads)
        v = self._gate("v", qkv[2], memory=memory_heads)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self._gate("attn_output", x, memory=memory_heads)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self._gate("proj_output", x, memory=memory)
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
