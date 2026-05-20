from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import math
import torch
from torch import Tensor, nn


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
        if self.mode == "multiplication":
            return x * gate
        if self.mode == "residual":
            return x + (x * gate)
        raise ValueError(f"Unsupported gated attention mode: {self.mode}")


def rotate_half(x: Tensor) -> Tensor:
    """LLaMA/Qwen-style rotate: [..., D] -> [..., D]. Requires even D."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Simple RoPE for tensors shaped [batch, heads, seq_len, head_dim]."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got head_dim={dim}")

        self.dim = dim
        self.base = float(base)
        self.max_position_embeddings = int(max_position_embeddings)

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor with shape [B, H, N, D]. Only device/dtype/B/N are used.
            position_ids: Optional tensor with shape [B, N] or [N]. If omitted, uses 0..N-1.
        Returns:
            cos, sin with shape [B, 1, N, D], broadcastable to q/k.
        """
        if x.dim() != 4:
            raise ValueError(f"RoPE expects x with shape [B, H, N, D], got {tuple(x.shape)}")

        B, _, N, _ = x.shape
        device = x.device

        if position_ids is None:
            position_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            position_ids = position_ids.to(device=device)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            if position_ids.shape != (B, N):
                raise ValueError(
                    f"position_ids must have shape [N] or [B, N], got {tuple(position_ids.shape)} for B={B}, N={N}"
                )

        # Compute freqs in fp32 for numerical stability, then cast back to x dtype.
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("bn,d->bnd", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype).unsqueeze(1)
        sin = emb.sin().to(dtype=x.dtype).unsqueeze(1)
        return cos, sin


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
        use_rope: bool = True,
        rope_base: float = 10000.0,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary_emb = (
            RotaryEmbedding(
                dim=self.head_dim,
                base=rope_base,
                max_position_embeddings=max_position_embeddings,
            )
            if use_rope
            else None
        )

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
            use_rope=bool(config.get("use_rope", config.get("rope", True))),
            rope_base=float(config.get("rope_base", config.get("rope_theta", 10000.0))),
            max_position_embeddings=int(
                config.get("max_position_embeddings", config.get("rope_max_position_embeddings", 2048))
            ),
        )
        gated.qkv.load_state_dict(attention.qkv.state_dict())
        gated.proj.load_state_dict(attention.proj.state_dict())
        return gated

    def _gate(self, position: str, x: Tensor) -> Tensor:
        gate = self.gates[position] if position in self.gates else None
        return gate(x) if gate is not None else x

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if attn_bias is not None:
            raise AssertionError("GatedSelfAttention does not support nested tensor attention bias")

        B, N, C = x.shape
        x = self._gate("pre_qkv", x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self._gate("q", qkv[0])
        k = self._gate("k", qkv[1])
        v = self._gate("v", qkv[2])

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, position_ids=position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scale exactly once. The previous version scaled q and the attention score.
        attn = q @ k.transpose(-2, -1) * self.inv_sqrt_head_dim
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
