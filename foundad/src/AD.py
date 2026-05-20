import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule
from src.masks import MultiBlockMaskGenerator
from src.utils.tensors import apply_masks, repeat_interleave_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        gated_attention=meta.get("gated_attention"),
        backbone_gating=meta.get("backbone_gating"),
        weights=meta.get("weights"),
    )


def _build_mask_generator(model: VisionModule, cfg: Dict[str, Any]) -> MultiBlockMaskGenerator:
    mask_cfg = cfg["meta"].get("ijepa_mask", {})
    return MultiBlockMaskGenerator(
        num_patches=model.num_patches,
        enc_mask_scale=mask_cfg.get("enc_mask_scale", (0.85, 1.0)),
        pred_mask_scale=mask_cfg.get("pred_mask_scale", (0.15, 0.2)),
        aspect_ratio=mask_cfg.get("aspect_ratio", (0.75, 1.5)),
        num_enc_masks=mask_cfg.get("num_enc_masks", 1),
        num_pred_masks=mask_cfg.get("num_pred_masks", 4),
        min_keep=mask_cfg.get("min_keep", 10),
        allow_overlap=mask_cfg.get("allow_overlap", False),
    )


def _predictor_mask_error(
    model: VisionModule,
    img: torch.Tensor,
    paths: List[str],
    n_layer: int,
    mask_generator: MultiBlockMaskGenerator,
    mask_rounds: int,
    context_mode: str = "gated",
    target_layer_norm: bool = True,
) -> torch.Tensor:
    h_raw = model.target_features(img, paths, n_layer=n_layer)
    h_target = F.layer_norm(h_raw, (h_raw.size(-1),)) if target_layer_norm else h_raw
    if context_mode == "raw":
        h_context = h_raw
    elif context_mode == "gated":
        h_context = model.apply_backbone_gate(h_raw)
    else:
        raise ValueError("testing.context_mode must be 'gated' or 'raw'")

    B, N, _ = h_raw.shape
    score_sum = h_raw.new_zeros((B, N))
    score_count = h_raw.new_zeros((B, N))

    for _ in range(mask_rounds):
        masks_enc, masks_pred = mask_generator(batch_size=B, device=img.device)
        target = apply_masks(h_target, masks_pred)
        target = repeat_interleave_batch(target, B, repeat=len(masks_enc))
        context = apply_masks(h_context, masks_enc)
        pred = model.predictor(context, masks_enc, masks_pred)
        err = F.mse_loss(pred, target, reduction="none").mean(dim=-1)

        offset = 0
        for pred_mask in masks_pred:
            for _enc_mask in masks_enc:
                err_block = err[offset : offset + B]
                for b in range(B):
                    score_sum[b].scatter_add_(0, pred_mask[b], err_block[b])
                    score_count[b].scatter_add_(0, pred_mask[b], torch.ones_like(err_block[b]))
                offset += B

    observed = score_count > 0
    scores = torch.zeros_like(score_sum)
    scores[observed] = score_sum[observed] / score_count[observed]
    if not bool(observed.all().item()):
        for b in range(B):
            fill = scores[b, observed[b]].mean() if bool(observed[b].any().item()) else scores.new_tensor(0.0)
            scores[b, ~observed[b]] = fill
    return scores

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    if state.get("backbone_gate") is not None and model.backbone_gate is not None:
        model.backbone_gate.load_state_dict(state["backbone_gate"])
    if "predictor" in state:
        model.predictor.load_state_dict(state["predictor"], strict=False)
    if model.projector is not None:
        projector_state = state.get("projector")
        if projector_state is not None:
            model.projector.load_state_dict(projector_state)
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    mask_generator = _build_mask_generator(model, cfg)
    mask_rounds = int(cfg.get("testing", {}).get("mask_rounds", 16))
    context_mode = str(cfg.get("testing", {}).get("context_mode", "gated")).lower()
    target_layer_norm = bool(cfg["meta"].get("ijepa_mask", {}).get("target_layer_norm", True))

    dataset_name = cfg["data"].get("dataset", "mvtec")
    if dataset_name == 'mvtec':
        classnames = cfg["data"]["mvtec_classnames"] 
        K = cfg["testing"]["K_top_mvtec"]
    elif dataset_name == 'visa':
        classnames = cfg["data"]["visa_classnames"]
        K = cfg["testing"]["K_top_visa"]
    else:
        raise NotImplementedError
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        print(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            l = _predictor_mask_error(
                model=model,
                img=img,
                paths=paths,
                n_layer=n_layer,
                mask_generator=mask_generator,
                mask_rounds=mask_rounds,
                context_mode=context_mode,
                target_layer_norm=target_layer_norm,
            )

            topk = torch.topk(l, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(img.cpu()); mask_buf.append(mask.cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    if state.get("backbone_gate") is not None and model.backbone_gate is not None:
        model.backbone_gate.load_state_dict(state["backbone_gate"])
    if "predictor" in state:
        model.predictor.load_state_dict(state["predictor"], strict=False)
    if model.projector is not None:
        projector_state = state.get("projector")
        if projector_state is not None:
            model.projector.load_state_dict(projector_state)
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    mask_generator = _build_mask_generator(model, cfg)
    mask_rounds = int(cfg.get("testing", {}).get("mask_rounds", 16))
    context_mode = str(cfg.get("testing", {}).get("context_mode", "gated")).lower()
    target_layer_norm = bool(cfg["meta"].get("ijepa_mask", {}).get("target_layer_norm", True))
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    assert dataset_name in cfg["data"]["test_root"] # check if eval on the same dataset the ckpt trained on
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        import cv2
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        l = _predictor_mask_error(
            model=model,
            img=img,
            paths=[str(path)],
            n_layer=n_layer,
            mask_generator=mask_generator,
            mask_rounds=mask_rounds,
            context_mode=context_mode,
            target_layer_norm=target_layer_norm,
        )

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        save_dir = (out_root / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()
