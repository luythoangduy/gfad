import math
import os
import random
from typing import List, Sequence, Tuple, Union

import PIL
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, distributed
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRotate90or270:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.choice([90, 270])
            return TF.rotate(img, angle)
        return img


def build_base_transform(resize: int = 518):
    return [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_train_transform(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
):
    ops = []

    if use_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))

    if use_vflip:
        ops.append(transforms.RandomVerticalFlip(p=0.2))

    if use_rotate90:
        ops.append(RandomRotate90or270(p=0.2))

    if use_color_jitter:
        ops.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)],
                p=0.2,
            )
        )

    if use_gray:
        ops.append(transforms.RandomGrayscale(p=0.1))

    if use_blur:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23 if resize >= 384 else 11, sigma=(0.1, 2.0))],
                p=0.2,
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)


def build_train_transform_new(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_any=0.3,
):
    candidates = []
    if use_hflip:
        candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:
        candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:
        candidates.append(RandomRotate90or270(p=1.0))
    if use_color_jitter:
        candidates.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.05))
    if use_gray:
        candidates.append(transforms.Lambda(lambda im: im.convert("L").convert("RGB")))
    if use_blur:
        candidates.append(transforms.GaussianBlur(kernel_size=23 if resize >= 384 else 11, sigma=(0.1, 2.0)))

    ops = []
    if candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(candidates)],
                p=p_any,
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)


def build_train_transform_staged(
    resize=518,
    use_hflip=False,
    use_vflip=False,
    use_rotate90=False,
    use_color_jitter=False,
    use_gray=False,
    use_blur=False,
    p_orient=0.3,
    p_appear=0.3,
):
    ops = []

    orient_candidates = []
    if use_hflip:
        orient_candidates.append(transforms.RandomHorizontalFlip(p=1.0))
    if use_vflip:
        orient_candidates.append(transforms.RandomVerticalFlip(p=1.0))
    if use_rotate90:
        orient_candidates.append(RandomRotate90or270(p=1.0))

    if orient_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(orient_candidates)],
                p=p_orient,
            )
        )

    appear_candidates = []
    if use_color_jitter:
        appear_candidates.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.05))
    if use_gray:
        appear_candidates.append(transforms.RandomGrayscale(p=1.0))
    if use_blur:
        kernel_size = 23 if resize >= 384 else 11
        appear_candidates.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)))

    if appear_candidates:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomChoice(appear_candidates)],
                p=p_appear,
            )
        )

    ops.extend(build_base_transform(resize))
    return transforms.Compose(ops)


class TrainDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, resize=518, **kwargs):
        super().__init__(os.path.join(root, "train"))
        self.resize = resize
        self.root = os.path.join(root, "train")
        self.transform = build_train_transform_staged(
            self.resize,
            use_hflip=kwargs.get("use_hflip", False),
            use_vflip=kwargs.get("use_vflip", False),
            use_rotate90=kwargs.get("use_rotate90", False),
            use_color_jitter=kwargs.get("use_color_jitter", False),
            use_gray=kwargs.get("use_gray", False),
            use_blur=kwargs.get("use_blur", False),
        )
        self.samples = [(path, self.classes[target]) for (path, target) in self.samples]
        print(f"Totally {len(self.samples)} will be trained..")

    def __getitem__(self, index):
        path_train, target = self.samples[index]
        image_train = self.loader(path_train).convert("RGB")
        image_train = self.transform(image_train)
        return image_train, target, path_train


class TestDataset(Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=518,
        datasetname="mvtec",
        **kwargs,
    ):
        super().__init__()
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.source = source
        self.classnames_to_use = [classname]
        self.datasetname = datasetname

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(build_base_transform(resize))
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
            ]
        )
        self.imagesize = (3, resize, resize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly not in ("good", "ok")),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "test")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, file_name) for file_name in anomaly_files
                ]
                if self.datasetname == "mvtec":
                    if anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, file_name) for file_name in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None
                elif self.datasetname == "visa":
                    if anomaly != "ok":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, file_name) for file_name in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["ok"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.datasetname == "mvtec":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i] if anomaly != "good" else None)
                    elif self.datasetname == "visa":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i] if anomaly != "ok" else None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def build_dataloader(
    mode: str,
    root: str,
    batch_size: int,
    pin_mem: bool = True,
    **kwargs,
):
    """Return (dataset, dataloader, sampler)."""
    if mode == "train":
        dataset = TrainDataset(root=root, **kwargs)
        sampler = distributed.DistributedSampler(dataset)
        drop_last = True
    elif mode == "test":
        dataset = TestDataset(source=root, **kwargs)
        sampler = None
        drop_last = False
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and mode == "test"),
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return dataset, dataloader, sampler


SubclassInput = Union[str, Sequence[str]]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_mean_std(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=dtype).view(3, 1, 1)
    return mean, std


def _imagenet_unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_mean_std(img.device, img.dtype)
    return torch.clamp(img * std + mean, 0.0, 1.0)


def _imagenet_normalize(img: torch.Tensor) -> torch.Tensor:
    mean, std = _imagenet_mean_std(img.device, img.dtype)
    return (img - mean) / std


def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _to_uint8_bins(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * 255.0, 0.0, 255.0).to(torch.long)


def _hist256_from_01(x: torch.Tensor) -> torch.Tensor:
    bins = _to_uint8_bins(x).reshape(-1)
    return torch.bincount(bins, minlength=256).to(dtype=torch.float32)


def _otsu_threshold_uint8(x: torch.Tensor) -> torch.Tensor:
    hist = _hist256_from_01(x)
    device = x.device

    total = hist.sum()
    if total.item() <= 0:
        return torch.zeros((), device=device, dtype=torch.long)

    idx = torch.arange(256, device=device, dtype=torch.float32)
    weight_b = torch.cumsum(hist, dim=0)
    weight_f = total - weight_b

    sum_b = torch.cumsum(hist * idx, dim=0)
    sum_total = sum_b[-1]

    mean_b = sum_b / weight_b.clamp_min(1.0)
    mean_f = (sum_total - sum_b) / weight_f.clamp_min(1.0)

    between = weight_b * weight_f * (mean_b - mean_f).pow(2)
    valid = (weight_b > 0) & (weight_f > 0)
    between = torch.where(valid, between, torch.full_like(between, -1.0))
    return torch.argmax(between).to(torch.long)


def _triangle_threshold_uint8(x: torch.Tensor) -> torch.Tensor:
    hist = _hist256_from_01(x)
    device = x.device

    nonzero = torch.nonzero(hist > 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.long)
    if nonzero.numel() == 1:
        return nonzero[0].to(torch.long)

    left = int(nonzero[0].item())
    right = int(nonzero[-1].item())
    peak = int(torch.argmax(hist).item())

    if left == right:
        return torch.tensor(left, device=device, dtype=torch.long)

    if (peak - left) < (right - peak):
        start, end = peak, right
    else:
        start, end = left, peak

    if start == end:
        return torch.tensor(peak, device=device, dtype=torch.long)

    xs = torch.arange(start, end + 1, device=device, dtype=torch.float32)
    ys = hist[start : end + 1]

    x1 = torch.tensor(float(start), device=device)
    y1 = hist[start]
    x2 = torch.tensor(float(end), device=device)
    y2 = hist[end]

    denom = torch.sqrt((y2 - y1).pow(2) + (x2 - x1).pow(2)).clamp_min(1e-12)
    distances = torch.abs((y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1) / denom
    return xs[torch.argmax(distances)].to(torch.long)


def _binary_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return _to_uint8_bins(x) > threshold


def _pad_for_same_kernel(k: int) -> Tuple[int, int, int, int]:
    before = (k - 1) // 2
    after = k // 2
    return before, after, before, after


def _binary_dilation(mask: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return mask.bool()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    x = F.pad(x, _pad_for_same_kernel(k), mode="constant", value=0.0)
    y = F.max_pool2d(x, kernel_size=k, stride=1)
    return y.squeeze(0).squeeze(0) > 0.5


def _binary_erosion(mask: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return mask.bool()
    inv = 1.0 - mask.float().unsqueeze(0).unsqueeze(0)
    inv = F.pad(inv, _pad_for_same_kernel(k), mode="constant", value=0.0)
    eroded_inv = F.max_pool2d(inv, kernel_size=k, stride=1)
    return (1.0 - eroded_inv.squeeze(0).squeeze(0)) > 0.5


def _binary_closing(mask: torch.Tensor, k: int) -> torch.Tensor:
    return _binary_erosion(_binary_dilation(mask, k), k)


def _binary_opening(mask: torch.Tensor, k: int) -> torch.Tensor:
    return _binary_dilation(_binary_erosion(mask, k), k)


def generate_target_foreground_mask(img: torch.Tensor, subclass: str) -> torch.Tensor:
    if not torch.is_tensor(img):
        raise TypeError("img must be a torch.Tensor")
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError("img must have shape [3, H, W]")

    _, h, w = img.shape
    device = img.device
    rgb = _imagenet_unnormalize(img)
    gray = _rgb_to_gray(rgb)

    if subclass in ["carpet", "leather", "tile", "wood", "cable", "transistor"]:
        target_foreground_mask = torch.ones((h, w), device=device, dtype=torch.bool)
    elif subclass == "pill":
        target_foreground_mask = _binary_threshold(gray, _otsu_threshold_uint8(gray))
    elif subclass in ["hazelnut", "metal_nut", "toothbrush"]:
        target_foreground_mask = _binary_threshold(gray, _triangle_threshold_uint8(gray))
    elif subclass in ["bottle", "capsule", "grid", "screw", "zipper"]:
        target_foreground_mask = ~_binary_threshold(gray, _otsu_threshold_uint8(gray))
    elif subclass in ["capsules"]:
        target_foreground_mask = torch.ones((h, w), device=device, dtype=torch.bool)
    elif subclass in ["pcb1", "pcb2", "pcb3", "pcb4"]:
        img_seg = rgb[2]
        target_foreground_mask = _binary_threshold(img_seg, _triangle_threshold_uint8(img_seg))
        target_foreground_mask = _binary_closing(target_foreground_mask, 8)
        target_foreground_mask = _binary_opening(target_foreground_mask, 3)
    elif subclass in ["candle", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pipe_fryum"]:
        target_foreground_mask = _binary_threshold(gray, _otsu_threshold_uint8(gray))
        target_foreground_mask = _binary_closing(target_foreground_mask, 3)
        target_foreground_mask = _binary_opening(target_foreground_mask, 3)
    elif subclass in ["bracket_black", "bracket_brown", "connector"]:
        img_seg = rgb[1]
        target_foreground_mask = ~_binary_threshold(img_seg, _otsu_threshold_uint8(img_seg))
    elif subclass in ["bracket_white", "tubes"]:
        img_seg = rgb[2]
        target_foreground_mask = _binary_threshold(img_seg, _otsu_threshold_uint8(img_seg))
    elif subclass in ["metal_plate"]:
        target_foreground_mask = ~_binary_threshold(gray, _otsu_threshold_uint8(gray))
    else:
        raise NotImplementedError(f"Unsupported foreground segmentation category: {subclass}")

    target_foreground_mask = _binary_closing(target_foreground_mask, 6)
    target_foreground_mask = _binary_opening(target_foreground_mask, 6)
    return target_foreground_mask.bool()


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[0], rgb[1], rgb[2]
    maxc = torch.max(rgb, dim=0).values
    minc = torch.min(rgb, dim=0).values
    delta = maxc - minc

    v = maxc
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-12), torch.zeros_like(maxc))
    h = torch.zeros_like(maxc)
    nonzero = delta > 1e-12

    hr = ((g - b) / delta.clamp_min(1e-12)) % 6.0
    hg = ((b - r) / delta.clamp_min(1e-12)) + 2.0
    hb = ((r - g) / delta.clamp_min(1e-12)) + 4.0

    h = torch.where((maxc == r) & nonzero, hr, h)
    h = torch.where((maxc == g) & nonzero, hg, h)
    h = torch.where((maxc == b) & nonzero, hb, h)
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=0)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[0], hsv[1], hsv[2]
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.long) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    zeros = torch.zeros_like(h)
    out = torch.stack([zeros, zeros, zeros], dim=0)
    masks = [i == k for k in range(6)]
    candidates = [
        torch.stack([v, t, p], dim=0),
        torch.stack([q, v, p], dim=0),
        torch.stack([p, v, t], dim=0),
        torch.stack([p, q, v], dim=0),
        torch.stack([t, p, v], dim=0),
        torch.stack([v, p, q], dim=0),
    ]

    for mask, candidate in zip(masks, candidates):
        out = torch.where(mask.unsqueeze(0), candidate, out)
    return out


class ColorJitterGPU:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1, hue: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def _sample_factor(device: torch.device, dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
        return torch.empty((), device=device, dtype=dtype).uniform_(low, high)

    def _adjust_brightness(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.brightness is None or self.brightness == 0:
            return rgb
        factor = self._sample_factor(rgb.device, rgb.dtype, max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
        return torch.clamp(rgb * factor, 0.0, 1.0)

    def _adjust_contrast(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.contrast is None or self.contrast == 0:
            return rgb
        factor = self._sample_factor(rgb.device, rgb.dtype, max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
        gray = _rgb_to_gray(rgb).mean()
        return torch.clamp((rgb - gray) * factor + gray, 0.0, 1.0)

    def _adjust_saturation(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.saturation is None or self.saturation == 0:
            return rgb
        factor = self._sample_factor(rgb.device, rgb.dtype, max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
        gray = _rgb_to_gray(rgb).unsqueeze(0)
        return torch.clamp((rgb - gray) * factor + gray, 0.0, 1.0)

    def _adjust_hue(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.hue is None or self.hue == 0:
            return rgb
        hue = min(float(self.hue), 0.5)
        shift = self._sample_factor(rgb.device, rgb.dtype, -hue, hue)
        hsv = _rgb_to_hsv(rgb)
        hsv[0] = (hsv[0] + shift) % 1.0
        return torch.clamp(_hsv_to_rgb(hsv), 0.0, 1.0)

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        rgb = _imagenet_unnormalize(patch)
        ops = [
            self._adjust_brightness,
            self._adjust_contrast,
            self._adjust_saturation,
            self._adjust_hue,
        ]
        random.shuffle(ops)
        for op in ops:
            rgb = op(rgb)
        return _imagenet_normalize(torch.clamp(rgb, 0.0, 1.0))


def _rotate_tensor_bilinear_expand(patch: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    if patch.ndim != 3:
        raise ValueError("patch must have shape [C, H, W]")

    _, h, w = patch.shape
    device = patch.device
    dtype = patch.dtype

    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    abs_cos = abs(cos_a)
    abs_sin = abs(sin_a)
    new_w = max(1, int(math.ceil(w * abs_cos + h * abs_sin)))
    new_h = max(1, int(math.ceil(h * abs_cos + w * abs_sin)))

    ys = torch.arange(new_h, device=device, dtype=dtype) - (new_h - 1) / 2.0
    xs = torch.arange(new_w, device=device, dtype=dtype) - (new_w - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x_src = cos_a * xx + sin_a * yy + (w - 1) / 2.0
    y_src = -sin_a * xx + cos_a * yy + (h - 1) / 2.0

    x_norm = 2.0 * x_src / (w - 1) - 1.0 if w > 1 else torch.zeros_like(x_src)
    y_norm = 2.0 * y_src / (h - 1) - 1.0 if h > 1 else torch.zeros_like(y_src)

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    rotated = F.grid_sample(
        patch.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.squeeze(0)


def _random_int(low: int, high_inclusive: int) -> int:
    if high_inclusive < low:
        raise ValueError(f"Invalid randint range [{low}, {high_inclusive}]")
    return random.randint(low, high_inclusive)


def _random_float(low: float, high: float) -> float:
    return random.uniform(low, high)


def _choose_valid_top_left(mask: torch.Tensor, patch_h: int, patch_w: int) -> Union[Tuple[int, int], None]:
    h, w = mask.shape
    max_y = h - patch_h
    max_x = w - patch_w
    if max_y < 0 or max_x < 0:
        return None

    valid_mask = mask[: max_y + 1, : max_x + 1]
    coords = torch.nonzero(valid_mask, as_tuple=False)
    if coords.numel() == 0:
        return None

    idx = torch.randint(0, coords.shape[0], (), device=mask.device)
    yx = coords[idx]
    return int(yx[0].item()), int(yx[1].item())


def _normalize_subclasses(subclasses: SubclassInput, batch_size: int) -> List[str]:
    if isinstance(subclasses, str):
        return [subclasses] * batch_size
    subclasses = list(subclasses)
    if len(subclasses) != batch_size:
        raise ValueError(f"Expected {batch_size} subclasses, got {len(subclasses)}")
    return subclasses


class CutPaste(object):
    def __init__(self, colorJitter=0.1):
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = ColorJitterGPU(
                brightness=colorJitter,
                contrast=colorJitter,
                saturation=colorJitter,
                hue=colorJitter,
            )

    def __call__(self, imgs):
        return imgs, imgs


class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02, 0.25], aspect_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, imgs, subclass):
        if not torch.is_tensor(imgs):
            raise TypeError("imgs must be a torch.Tensor")
        if imgs.ndim != 4:
            raise ValueError("imgs must have shape [B, C, H, W]")

        batch_size = imgs.shape[0]
        subclasses = _normalize_subclasses(subclass, batch_size)
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            augmented_imgs[i] = self.process_image(imgs[i], subclasses[i])

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape
        target_foreground_mask = generate_target_foreground_mask(img, subclass)

        area = h * w
        target_area = _random_float(self.area_ratio[0], self.area_ratio[1]) * area
        aspect_ratio = _random_float(self.aspect_ratio, 1.0 / self.aspect_ratio)

        cut_w = int(round(math.sqrt(target_area * aspect_ratio)))
        cut_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if cut_w <= 0 or cut_h <= 0 or cut_w > w or cut_h > h:
            return img

        from_x = _random_int(0, w - cut_w)
        from_y = _random_int(0, h - cut_h)
        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        valid_pos = _choose_valid_top_left(target_foreground_mask, cut_h, cut_w)
        if valid_pos is None:
            return img

        to_y, to_x = valid_pos
        augmented = img.clone()
        augmented[:, to_y : to_y + cut_h, to_x : to_x + cut_w] = patch
        return augmented


class CutPasteScar(CutPaste):
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, imgs, subclass):
        if not torch.is_tensor(imgs):
            raise TypeError("imgs must be a torch.Tensor")
        if imgs.ndim != 4:
            raise ValueError("imgs must have shape [B, C, H, W]")

        batch_size = imgs.shape[0]
        subclasses = _normalize_subclasses(subclass, batch_size)
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            augmented_imgs[i] = self.process_image(imgs[i], subclasses[i])

        return imgs, augmented_imgs

    def process_image(self, img, subclass):
        img = img.clone()
        _, h, w = img.shape
        target_foreground_mask = generate_target_foreground_mask(img, subclass)

        cut_w = int(_random_float(float(self.width[0]), float(self.width[1])))
        cut_h = int(_random_float(float(self.height[0]), float(self.height[1])))
        if cut_w <= 0 or cut_h <= 0 or cut_w > w or cut_h > h:
            return img

        from_x = _random_int(0, w - cut_w)
        from_y = _random_int(0, h - cut_h)
        patch = img[:, from_y : from_y + cut_h, from_x : from_x + cut_w]

        if self.colorJitter is not None:
            patch = self.colorJitter(patch)

        rot_deg = _random_float(float(self.rotation[0]), float(self.rotation[1]))
        patch = _rotate_tensor_bilinear_expand(patch, angle_degrees=rot_deg)

        _, patch_h, patch_w = patch.shape
        if patch_w > w or patch_h > h:
            return img

        valid_pos = _choose_valid_top_left(target_foreground_mask, patch_h, patch_w)
        if valid_pos is None:
            return img

        to_y, to_x = valid_pos
        augmented = img.clone()
        mask = torch.ones_like(patch, device=patch.device, dtype=patch.dtype)
        return self.paste_with_mask(augmented, patch, mask, to_y, to_x)

    def paste_with_mask(self, img, patch, mask, top, left):
        _, h, w = img.shape
        _, patch_h, patch_w = patch.shape
        if top + patch_h > h or left + patch_w > w:
            return img

        img_region = img[:, top : top + patch_h, left : left + patch_w]
        mask = mask.to(device=img_region.device, dtype=img_region.dtype)
        patch = patch.to(device=img_region.device, dtype=img_region.dtype)
        img_region = img_region * (1.0 - mask) + patch * mask
        img[:, top : top + patch_h, left : left + patch_w] = img_region
        return img


class CutPasteUnion(object):
    def __init__(self, **kwargs):
        self.cutpaste_normal = CutPasteNormal(**kwargs)
        self.cutpaste_scar = CutPasteScar(**kwargs)

    def __call__(self, imgs, subclasses):
        if not torch.is_tensor(imgs):
            raise TypeError("imgs must be a torch.Tensor")
        if imgs.ndim != 4:
            raise ValueError("imgs must have shape [B, C, H, W]")

        batch_size = imgs.shape[0]
        subclasses = _normalize_subclasses(subclasses, batch_size)
        augmented_imgs = imgs.clone()

        for i in range(batch_size):
            img = imgs[i].unsqueeze(0)
            subclass = subclasses[i]
            if random.random() < 0.5:
                _, augmented = self.cutpaste_normal(img, subclass)
            else:
                _, augmented = self.cutpaste_scar(img, subclass)
            augmented_imgs[i] = augmented.squeeze(0)

        return imgs, augmented_imgs
