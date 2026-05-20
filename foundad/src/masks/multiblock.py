from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


class MultiBlockMaskGenerator:
    """I-JEPA-style rectangular context/target patch mask sampler."""

    def __init__(
        self,
        num_patches: int,
        enc_mask_scale: Sequence[float] = (0.85, 1.0),
        pred_mask_scale: Sequence[float] = (0.15, 0.2),
        aspect_ratio: Sequence[float] = (0.75, 1.5),
        num_enc_masks: int = 1,
        num_pred_masks: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ) -> None:
        side = int(num_patches**0.5)
        if side * side != num_patches:
            raise ValueError(f"Multi-block masks require a square patch grid, got {num_patches} patches")
        self.height = side
        self.width = side
        self.num_patches = num_patches
        self.enc_mask_scale = tuple(enc_mask_scale)
        self.pred_mask_scale = tuple(pred_mask_scale)
        self.aspect_ratio = tuple(aspect_ratio)
        self.num_enc_masks = int(num_enc_masks)
        self.num_pred_masks = int(num_pred_masks)
        self.min_keep = int(min_keep)
        self.allow_overlap = bool(allow_overlap)

    def _sample_block_size(self, scale: Tuple[float, float], aspect_ratio: Tuple[float, float]) -> Tuple[int, int]:
        mask_scale = torch.empty(1).uniform_(float(scale[0]), float(scale[1])).item()
        ar = torch.empty(1).uniform_(float(aspect_ratio[0]), float(aspect_ratio[1])).item()
        max_keep = max(1, int(self.height * self.width * mask_scale))
        h = max(1, int(round(math.sqrt(max_keep * ar))))
        w = max(1, int(round(math.sqrt(max_keep / ar))))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return h, w

    def _sample_block_mask(
        self,
        block_size: Tuple[int, int],
        acceptable_regions: Optional[Iterable[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = block_size
        acceptable_regions = list(acceptable_regions or [])
        tries = 0
        timeout = 20

        while True:
            top = torch.randint(0, self.height - h + 1, (1,)).item()
            left = torch.randint(0, self.width - w + 1, (1,)).item()
            region = torch.zeros((self.height, self.width), dtype=torch.int32)
            region[top : top + h, left : left + w] = 1

            mask = region.clone()
            for acceptable in acceptable_regions[: max(len(acceptable_regions) - tries, 0)]:
                mask *= acceptable
            mask = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
            if len(mask) > self.min_keep:
                complement = torch.ones((self.height, self.width), dtype=torch.int32)
                complement[top : top + h, left : left + w] = 0
                return mask, complement

            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = 20

    def __call__(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pred_size = self._sample_block_size(self.pred_mask_scale, self.aspect_ratio)
        enc_size = self._sample_block_size(self.enc_mask_scale, (1.0, 1.0))

        masks_pred_per_image: List[List[torch.Tensor]] = []
        masks_enc_per_image: List[List[torch.Tensor]] = []
        min_keep_pred = self.num_patches
        min_keep_enc = self.num_patches

        for _ in range(batch_size):
            masks_p, complements = [], []
            for _ in range(self.num_pred_masks):
                mask, complement = self._sample_block_mask(pred_size)
                masks_p.append(mask)
                complements.append(complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            masks_pred_per_image.append(masks_p)

            acceptable_regions = None if self.allow_overlap else complements
            masks_e = []
            for _ in range(self.num_enc_masks):
                mask, _ = self._sample_block_mask(enc_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            masks_enc_per_image.append(masks_e)

        masks_pred = [
            torch.stack([masks_pred_per_image[b][i][:min_keep_pred] for b in range(batch_size)], dim=0)
            for i in range(self.num_pred_masks)
        ]
        masks_enc = [
            torch.stack([masks_enc_per_image[b][i][:min_keep_enc] for b in range(batch_size)], dim=0)
            for i in range(self.num_enc_masks)
        ]
        if device is not None:
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
        return masks_enc, masks_pred
