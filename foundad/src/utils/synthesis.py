
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import morphology
import cv2


def generate_target_foreground_mask(img: np.ndarray, subclass: str) -> np.ndarray:
    inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    img_tensor = inv_normalize(img)

    img_tensor = torch.clamp(img_tensor, 0, 1)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    img_np_uint8 = (img_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2BGR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass == 'pill':
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
        _, target_foreground_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        target_foreground_mask = (target_foreground_mask > 0).astype(int)
    elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
        _, target_background_mask = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        target_background_mask = (target_background_mask > 0).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['capsules']:
        target_foreground_mask = np.ones_like(img_gray)
    elif subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
        _, target_foreground_mask = cv2.threshold(img_np_uint8[:, :, 2], 100, 255,
                                                    cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(8))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pipe_fryum']:
        _, target_foreground_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_foreground_mask = target_foreground_mask.astype(bool).astype(int)
        target_foreground_mask = morphology.closing(target_foreground_mask, morphology.square(3))
        target_foreground_mask = morphology.opening(target_foreground_mask, morphology.square(3))
    elif subclass in ['bracket_black', 'bracket_brown', 'connector']:
        img_seg = img_np_uint8[:, :, 1]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    elif subclass in ['bracket_white', 'tubes']:
        img_seg = img_np_uint8[:, :, 2]
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = target_background_mask
    elif subclass in ['metal_plate']:
        img_seg = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2GRAY)
        _, target_background_mask = cv2.threshold(img_seg, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        target_background_mask = target_background_mask.astype(bool).astype(int)
        target_foreground_mask = 1 - target_background_mask
    else:
        raise NotImplementedError("Unsupported foreground segmentation category")

    target_foreground_mask = morphology.closing(
        target_foreground_mask, morphology.square(6))
    target_foreground_mask = morphology.opening(
        target_foreground_mask, morphology.square(6))

    return target_foreground_mask


def _binary_dilation(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    pad_before = (kernel_size - 1) // 2
    pad_after = kernel_size // 2
    x = F.pad(x, (pad_before, pad_after, pad_before, pad_after), value=0.0)
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1).squeeze(0).squeeze(0) > 0


def _binary_erosion(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x = mask.float().view(1, 1, *mask.shape)
    pad_before = (kernel_size - 1) // 2
    pad_after = kernel_size // 2
    x = F.pad(x, (pad_before, pad_after, pad_before, pad_after), value=1.0)
    eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1)
    return eroded.squeeze(0).squeeze(0) > 0.5


def _binary_closing(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return _binary_erosion(_binary_dilation(mask, kernel_size), kernel_size)


def _binary_opening(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return _binary_dilation(_binary_erosion(mask, kernel_size), kernel_size)


def _torch_to_cupy(tensor: torch.Tensor):
    import cupy as cp
    from torch.utils import dlpack

    tensor = tensor.contiguous()
    try:
        return cp.from_dlpack(tensor)
    except TypeError:
        return cp.fromDlpack(dlpack.to_dlpack(tensor))


def _cupy_to_torch(array):
    import cupy as cp
    from torch.utils import dlpack

    array = cp.ascontiguousarray(array)
    try:
        return torch.from_dlpack(array)
    except TypeError:
        return dlpack.from_dlpack(array.toDlpack())


def generate_target_foreground_mask_gpu(img: torch.Tensor, subclass: str):
    """Return a CUDA bool mask [H, W], or None when GPU thresholding is unavailable."""
    if not isinstance(img, torch.Tensor) or not img.is_cuda:
        return None

    try:
        from cucim.skimage.filters import threshold_otsu, threshold_triangle
    except Exception:
        return None

    mean = img.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_01 = (img * std + mean).clamp(0, 1)
    img_uint8 = (img_01 * 255).to(torch.uint8)
    gray = (
        0.299 * img_uint8[0].float()
        + 0.587 * img_uint8[1].float()
        + 0.114 * img_uint8[2].float()
    ).round().clamp(0, 255).to(torch.uint8)

    try:
        gray_cp = _torch_to_cupy(gray)
        if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor', 'capsules']:
            target_foreground_mask = torch.ones_like(gray, dtype=torch.bool)
        elif subclass == 'pill':
            threshold = threshold_otsu(gray_cp)
            target_foreground_mask = _cupy_to_torch(gray_cp > threshold).to(device=img.device, dtype=torch.bool)
        elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
            threshold = threshold_triangle(gray_cp)
            target_foreground_mask = _cupy_to_torch(gray_cp > threshold).to(device=img.device, dtype=torch.bool)
        elif subclass in ['bottle', 'capsule', 'grid', 'screw', 'zipper']:
            threshold = threshold_otsu(gray_cp)
            target_background_mask = _cupy_to_torch(gray_cp > threshold).to(device=img.device, dtype=torch.bool)
            target_foreground_mask = ~target_background_mask
        elif subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
            channel_cp = _torch_to_cupy(img_uint8[2])
            threshold = threshold_triangle(channel_cp)
            target_foreground_mask = _cupy_to_torch(channel_cp > threshold).to(device=img.device, dtype=torch.bool)
            target_foreground_mask = _binary_closing(target_foreground_mask, 8)
            target_foreground_mask = _binary_opening(target_foreground_mask, 3)
        elif subclass in ['candle', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pipe_fryum']:
            threshold = threshold_otsu(gray_cp)
            target_foreground_mask = _cupy_to_torch(gray_cp > threshold).to(device=img.device, dtype=torch.bool)
            target_foreground_mask = _binary_closing(target_foreground_mask, 3)
            target_foreground_mask = _binary_opening(target_foreground_mask, 3)
        elif subclass in ['bracket_black', 'bracket_brown', 'connector']:
            channel_cp = _torch_to_cupy(img_uint8[1])
            threshold = threshold_otsu(channel_cp)
            target_background_mask = _cupy_to_torch(channel_cp > threshold).to(device=img.device, dtype=torch.bool)
            target_foreground_mask = ~target_background_mask
        elif subclass in ['bracket_white', 'tubes']:
            channel_cp = _torch_to_cupy(img_uint8[2])
            threshold = threshold_otsu(channel_cp)
            target_foreground_mask = _cupy_to_torch(channel_cp > threshold).to(device=img.device, dtype=torch.bool)
        elif subclass in ['metal_plate']:
            threshold = threshold_otsu(gray_cp)
            target_background_mask = _cupy_to_torch(gray_cp > threshold).to(device=img.device, dtype=torch.bool)
            target_foreground_mask = ~target_background_mask
        else:
            return None
    except Exception:
        return None

    target_foreground_mask = _binary_closing(target_foreground_mask, 6)
    target_foreground_mask = _binary_opening(target_foreground_mask, 6)
    return target_foreground_mask

class CutPaste(object):
    def __init__(self, colorJitter=0.1, use_gpu_mask=True):
        self.use_gpu_mask = use_gpu_mask
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter)

    def __call__(self, imgs):
        return imgs, imgs

    def _foreground_mask(self, img, subclass):
        if self.use_gpu_mask:
            mask = generate_target_foreground_mask_gpu(img, subclass)
            if mask is not None:
                return mask
        return generate_target_foreground_mask(img, subclass)

    @staticmethod
    def _choose_location(target_foreground_mask, patch_h, patch_w, h, w):
        if isinstance(target_foreground_mask, torch.Tensor):
            mask_indices = torch.nonzero(target_foreground_mask, as_tuple=False)
            if mask_indices.numel() == 0:
                return None

            valid = mask_indices[
                (mask_indices[:, 0] + patch_h <= h)
                & (mask_indices[:, 1] + patch_w <= w)
            ]
            if valid.numel() == 0:
                return None

            idx = torch.randint(valid.size(0), (1,), device=valid.device)
            yx = valid[idx].squeeze(0)
            return int(yx[0].item()), int(yx[1].item())

        mask_indices = np.argwhere(target_foreground_mask == 1)
        if len(mask_indices) == 0:
            return None

        valid_indices = []
        for y, x in mask_indices:
            if y + patch_h <= h and x + patch_w <= w:
                valid_indices.append((y, x))

        if len(valid_indices) == 0:
            return None

        return random.choice(valid_indices)

class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = self._foreground_mask(img, subclass)  # [H, W]


        area = h * w
        target_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = random.uniform(self.aspect_ratio, 1 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        loc = self._choose_location(target_foreground_mask, cut_h, cut_w, h, w)
        if loc is None:
            return img  

        to_y, to_x = loc

        augmented = img.clone()
        augmented[:, to_y:to_y+cut_h, to_x:to_x+cut_w] = patch

        return augmented

class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass):
        batch_size, _, h, w = imgs.shape
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i]
            augmented = self.process_image(img, subclass)
            augmented_imgs[i] = augmented

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape

        target_foreground_mask = self._foreground_mask(img, subclass)
    
        cut_w = int(random.uniform(*self.width))
        cut_h = int(random.uniform(*self.height))

        if cut_w <= 0 or cut_h <= 0:
            return img

        from_x = random.randint(0, w - cut_w)
        from_y = random.randint(0, h - cut_h)

        patch = img[:, from_y:from_y+cut_h, from_x:from_x+cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = random.uniform(*self.rotation)
        patch = TF.rotate(patch, angle=rot_deg, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

        _, patch_h, patch_w = patch.shape

        to_x = random.randint(0, w - patch_w)
        to_y = random.randint(0, h - patch_h)

        loc = self._choose_location(target_foreground_mask, patch_h, patch_w, h, w)
        if loc is None:
            return img  

        to_y, to_x = loc

        augmented = img.clone()
        mask = torch.ones_like(patch)
        augmented = self.paste_with_mask(augmented, patch, mask, to_y, to_x)

        return augmented

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape

        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top:top+patch_h, left:left+patch_w]
        mask = mask.to(img_region.device)
        img_region = img_region * (1 - mask) + patch * mask
        img[:, top:top+patch_h, left:left+patch_w] = img_region

        return img

class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses):
        batch_size = imgs.shape[0]
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)  # [1, C, H, W]
            subclass = subclasses[i]
            if random.random() < 0.5:
                _, augmented = self.cutpaste_normal(img, subclass)
            else:
                _, augmented = self.cutpaste_scar(img, subclass)
            augmented_imgs[i] = augmented.squeeze(0)

        return imgs, augmented_imgs
