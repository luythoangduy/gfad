# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Neighbor-masked cross-attention predictor.

Predicts encoder features for patch i from context patches that exclude
patch i and its spatial neighborhood, so the predictor cannot fall back to
an identity shortcut (copying z_i or a near-duplicate neighbor into the
prediction).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dinov2.layers import Mlp
from src.dinov2.models.vision_transformer import get_2d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


def build_neighbor_mask(num_patches: int, mask_radius: int) -> torch.Tensor:
    """Boolean mask [N, N]; True at (i, j) means context patch j is blocked
    when reconstructing patch i (j lies within the Chebyshev ``mask_radius``
    neighborhood of i, which always includes i itself)."""
    grid_size = math.isqrt(num_patches)
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"NeighborMaskedCrossAttentionPredictor requires a square token grid; "
            f"got num_patches={num_patches}, which is not a perfect square."
        )
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)  # [N, 2] -> (row, col)

    row_diff = (coords[:, None, 0] - coords[None, :, 0]).abs()
    col_diff = (coords[:, None, 1] - coords[None, :, 1]).abs()
    chebyshev = torch.maximum(row_diff, col_diff)
    return chebyshev <= mask_radius


class NeighborMaskedCrossAttention(nn.Module):
    """Cross-attention: queries attend to context tokens, with attention
    scores for masked (query, context) pairs replaced by a constant before
    softmax so no gradient flows back through those context tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = query.shape

        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        # masked_fill replaces (not biases) blocked scores with a constant,
        # so masked context tokens get exactly zero gradient, not just a
        # near-zero softmax weight.
        mask_value = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(neighbor_mask, mask_value)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class NeighborMaskedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm_query = norm_layer(dim)
        self.norm_context = norm_layer(dim)
        self.attn = NeighborMaskedCrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=proj_drop,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        query = query + self.attn(self.norm_query(query), self.norm_context(context), neighbor_mask)
        query = query + self.mlp(self.norm_mlp(query))
        return query


class NeighborMaskedCrossAttentionPredictor(nn.Module):
    """Reconstructs encoder tokens z: [B, N, embed_dim] -> [B, N, embed_dim]
    without letting prediction_i depend on z_i or any encoder token in its
    ``mask_radius`` spatial neighborhood.

    - Context stream: per-token Linear + LayerNorm on z (no token mixing).
    - Query stream: a learned mask token + fixed 2D sincos position embedding
      (no dependence on z).
    - Cross-attention blocks: queries attend to context under a neighbor
      mask; residuals only accumulate in the query stream.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_radius: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_query_pos: bool = True,
        qkv_bias: bool = True,
        feat_normed: bool = False,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()
        # `num_patches` is only a construction-time hint (e.g. for logging or
        # eager weight init order); some encoders report a default-resolution
        # patch count here that does not match the token count actually
        # produced at the configured crop_size. The real grid geometry is
        # always (re)derived from the runtime N in `forward`, never assumed.
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.mask_radius = mask_radius
        self.use_query_pos = use_query_pos
        self.init_std = init_std
        self.feat_normed = feat_normed

        # -- context stream (token-independent projection only)
        self.context_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.context_norm = norm_layer(predictor_embed_dim)

        # -- query stream (never derived from z)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Placeholders; `_build_geometry` fills these in with the correct
        # size, dtype and device the first time `forward` sees a given N.
        self.register_buffer("query_pos_embed", torch.zeros(1, 1, predictor_embed_dim), persistent=False)
        self.register_buffer("neighbor_mask", torch.zeros(1, 1, dtype=torch.bool), persistent=False)
        self._geometry_N: Optional[int] = None

        self.blocks = nn.ModuleList(
            [
                NeighborMaskedCrossAttentionBlock(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.apply(self._init_weights)
        trunc_normal_(self.mask_token, std=self.init_std)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_geometry(self, N: int) -> None:
        """(Re)build the fixed position embedding and neighbor mask for a
        token grid of size N, on whatever device the module currently lives
        on. Both buffers are non-persistent (derived, not learned), so this
        can safely run again whenever N changes across calls."""
        grid_size = math.isqrt(N)
        device = self.mask_token.device

        if self.use_query_pos:
            pos_embed = get_2d_sincos_pos_embed(self.predictor_embed_dim, grid_size)
            self.query_pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(device=device)

        self.neighbor_mask = build_neighbor_mask(N, self.mask_radius).to(device=device)
        self._geometry_N = N
        self.num_patches = N

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, _ = z.shape

        grid_size = math.isqrt(N)
        if grid_size * grid_size != N:
            raise ValueError(
                f"NeighborMaskedCrossAttentionPredictor requires a square token grid; "
                f"got N={N}, which is not a perfect square."
            )
        if self._geometry_N != N:
            self._build_geometry(N)

        # -- context stream: independent per-token projection of z, no mixing
        context = self.context_norm(self.context_embed(z))

        # -- query stream: mask token + fixed position embedding, never z
        query = self.mask_token.expand(B, N, self.predictor_embed_dim)
        if self.use_query_pos:
            query = query + self.query_pos_embed.to(dtype=query.dtype)

        neighbor_mask = self.neighbor_mask
        for blk in self.blocks:
            query = blk(query, context, neighbor_mask)

        query = self.predictor_norm(query)
        prediction = self.predictor_proj(query)

        if self.feat_normed:
            prediction = F.normalize(prediction, dim=-1)

        return prediction
