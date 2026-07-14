"""Sanity checks for NeighborMaskedCrossAttentionPredictor.

Run with: python -m scripts.verify_neighbor_masked_predictor
(On CPU-only machines, pin BLAS threads to avoid oversubscription stalls:
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.verify_neighbor_masked_predictor)

Checks:
  1. Output shape matches [B, N, encoder_dim].
  2. Forward output and backward gradients are finite (no NaN/Inf).
  3. Critical leakage check: d(prediction_i)/d(z_j) == 0 for every j in the
     mask_radius neighborhood of i (including j == i), and is generally
     nonzero for at least one j outside that neighborhood.
  4. Runtime token count N is re-derived per call, tolerating a stale/wrong
     construction-time num_patches hint.
  5. mask_radius validation: invalid or fully-blanketing radii raise instead
     of letting softmax silently degenerate to a uniform (leaking)
     distribution over masked tokens.
"""

import math

import torch

from src.neighbor_masked_predictor import NeighborMaskedCrossAttentionPredictor, build_neighbor_mask


def check_shape_and_finiteness():
    torch.manual_seed(0)
    B, grid, embed_dim, pred_dim, depth, heads = 2, 8, 32, 64, 3, 4
    N = grid * grid

    predictor = NeighborMaskedCrossAttentionPredictor(
        num_patches=N, embed_dim=embed_dim, predictor_embed_dim=pred_dim,
        depth=depth, num_heads=heads, mask_radius=1,
    )
    z = torch.randn(B, N, embed_dim, requires_grad=True)

    pred = predictor(z)
    assert pred.shape == (B, N, embed_dim), f"bad shape: {pred.shape}"
    assert torch.isfinite(pred).all(), "prediction contains NaN/Inf"

    loss = pred.pow(2).sum()
    loss.backward()

    assert z.grad is not None and torch.isfinite(z.grad).all(), "z.grad has NaN/Inf"
    for name, p in predictor.named_parameters():
        assert p.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"

    print(f"[OK] shape={tuple(pred.shape)}, forward/backward finite, all params received gradients")


def check_no_leakage():
    torch.manual_seed(1)
    B, grid, embed_dim, pred_dim, depth, heads, radius = 1, 6, 16, 32, 2, 4, 1
    N = grid * grid

    predictor = NeighborMaskedCrossAttentionPredictor(
        num_patches=N, embed_dim=embed_dim, predictor_embed_dim=pred_dim,
        depth=depth, num_heads=heads, mask_radius=radius,
    )
    neighbor_mask = build_neighbor_mask(N, radius)  # [N, N] True == blocked

    for i in [0, grid // 2 * grid + grid // 2, N - 1]:  # corner, center, corner
        z = torch.randn(B, N, embed_dim, requires_grad=True)
        pred = predictor(z)
        scalar = pred[:, i, :].sum()
        (grad,) = torch.autograd.grad(scalar, z, retain_graph=False)  # [B, N, embed_dim]

        per_token_grad = grad[0].abs().sum(dim=-1)  # [N]
        blocked = neighbor_mask[i]  # [N] True == must be zero-grad
        leaked = per_token_grad[blocked]
        assert torch.all(leaked == 0), (
            f"leakage detected for patch {i}: nonzero grad on masked tokens "
            f"{blocked.nonzero().flatten().tolist()} -> {leaked.tolist()}"
        )

        visible = per_token_grad[~blocked]
        assert torch.any(visible > 0), f"no gradient reached any visible context token for patch {i}"

    print(f"[OK] no-leakage check passed for corner/center patches (mask_radius={radius})")


def check_num_patches_mismatch_is_tolerated():
    """Some encoders report a construction-time num_patches (e.g. a hub
    model's default-resolution patch count) that differs from the token
    count actually produced at the configured crop_size. The predictor must
    not hard-fail in that case; it should re-derive geometry from the real
    runtime N instead."""
    torch.manual_seed(2)
    embed_dim, pred_dim, depth, heads = 8, 16, 2, 2
    predictor = NeighborMaskedCrossAttentionPredictor(
        num_patches=196, embed_dim=embed_dim, predictor_embed_dim=pred_dim,
        depth=depth, num_heads=heads, mask_radius=1,
    )
    B, N = 2, 64  # 8x8 grid, deliberately different from the num_patches=196 hint
    z = torch.randn(B, N, embed_dim, requires_grad=True)
    pred = predictor(z)
    assert pred.shape == (B, N, embed_dim)
    assert torch.isfinite(pred).all()
    pred.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    # switching back to a different N again (e.g. eval-time resolution change)
    z2 = torch.randn(1, 16, embed_dim)  # 4x4 grid
    pred2 = predictor(z2)
    assert pred2.shape == (1, 16, embed_dim)
    assert torch.isfinite(pred2).all()

    print("[OK] predictor tolerates num_patches hint != actual runtime N, and re-derives geometry per call")


def check_mask_radius_validation():
    """A mask_radius that is invalid, or large enough to block every context
    token for some query, must raise instead of silently letting softmax
    degenerate to a uniform distribution over masked (leaking) tokens."""
    for bad_radius in [-1, 1.5, "1"]:
        try:
            NeighborMaskedCrossAttentionPredictor(
                num_patches=16, embed_dim=8, predictor_embed_dim=8, depth=1, num_heads=2,
                mask_radius=bad_radius,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for mask_radius={bad_radius!r}")

    # 4x4 grid (N=16): mask_radius=10 covers every token from every query.
    predictor = NeighborMaskedCrossAttentionPredictor(
        num_patches=16, embed_dim=8, predictor_embed_dim=8, depth=1, num_heads=2, mask_radius=10,
    )
    z = torch.randn(1, 16, 8)
    try:
        predictor(z)
    except ValueError as e:
        assert "masks every context token" in str(e), f"unexpected error message: {e}"
    else:
        raise AssertionError("expected ValueError when mask_radius blanks out all context for a query")

    print("[OK] mask_radius validation rejects invalid/degenerate radii instead of silently leaking")


def check_mask_shape():
    for N, radius in [(16, 1), (64, 0), (1024, 1)]:
        grid = math.isqrt(N)
        mask = build_neighbor_mask(N, radius)
        assert mask.shape == (N, N)
        assert mask.dtype == torch.bool
        assert mask.diagonal().all(), "self must always be masked"
    print("[OK] neighbor mask shape/self-masking checks passed")


if __name__ == "__main__":
    check_shape_and_finiteness()
    check_no_leakage()
    check_num_patches_mismatch_is_tolerated()
    check_mask_radius_validation()
    check_mask_shape()
    print("All checks passed.")
