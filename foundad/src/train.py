
from __future__ import annotations

import os, sys, random, logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import yaml, numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.dataset import build_dataloader
from src.foundad import VisionModule
from src.masks import MultiBlockMaskGenerator
from src.utils.tensors import apply_masks, repeat_interleave_batch

_GLOBAL_SEED = 0
random.seed(42); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args: Dict[str, Any]):
        # ---------- basic ----------
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # ---------- model ----------
        mcfg = args["meta"]
        self.model = VisionModule(
            mcfg["model"],
            mcfg["pred_depth"],
            mcfg["pred_emb_dim"],
            if_pe=mcfg.get("if_pred_pe", True),
            feat_normed=mcfg.get("feat_normed", False),
            gated_attention=mcfg.get("gated_attention"),
            backbone_gating=mcfg.get("backbone_gating"),
            backbone_lora=mcfg.get("backbone_lora"),
            weights=mcfg.get("weights"),
            crop_size=mcfg.get("crop_size"),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.requires_grad_(False)
        self.model.predictor.requires_grad_(True)
        if self.model.backbone_lora is not None:
            self.model.backbone_lora.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")

        mask_cfg = mcfg.get("ijepa_mask", {})
        self.target_layer_norm = bool(mask_cfg.get("target_layer_norm", True))
        self.mask_generator = MultiBlockMaskGenerator(
            num_patches=self.model.num_patches,
            enc_mask_scale=mask_cfg.get("enc_mask_scale", (0.85, 1.0)),
            pred_mask_scale=mask_cfg.get("pred_mask_scale", (0.15, 0.2)),
            aspect_ratio=mask_cfg.get("aspect_ratio", (0.75, 1.5)),
            num_enc_masks=mask_cfg.get("num_enc_masks", 1),
            num_pred_masks=mask_cfg.get("num_pred_masks", 4),
            min_keep=mask_cfg.get("min_keep", 10),
            allow_overlap=mask_cfg.get("allow_overlap", False),
        )

        # ---------- data ----------
        dcfg = args["data"]
        assert dcfg["dataset"] in dcfg["data_name"] # check if the dataset aligns with the few-shot folder
        _, self.loader, self.sampler = build_dataloader(
            mode="train",
            root=dcfg["train_root"],
            batch_size=dcfg["batch_size"],
            num_workers=dcfg.get("num_workers", 0),
            pin_mem=dcfg["pin_mem"],
            resize=mcfg["crop_size"],
            use_hflip=dcfg.get("use_hflip",False),
            use_vflip=dcfg.get("use_vflip",False),
            use_rotate90=dcfg.get("use_rotate90",False),
            use_color_jitter=dcfg.get("use_color_jitter",False),
            use_gray=dcfg.get("use_gray",False),
            use_blur=dcfg.get("use_blur",False),
        )
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=[self.model.predictor, self.model.backbone_lora],
            wd=float(ocfg["weight_decay"]),
            lr=ocfg["lr"],
            lr_config=ocfg.get("lr_config", "const"),
            max_epoch=ocfg["epochs"],                         # for cosine_warmup
            min_lr=ocfg.get("min_lr", 1e-6),                  # for cosine_warmup
            warmup_epoch=ocfg.get("warmup_epoch", 5),         # for cosine_warmup
            step_size=ocfg.get("step_size", 300),             # for step
            gamma=ocfg.get("gamma", 0.1),                     # for step
        )
        self.epochs = ocfg["epochs"]
        self.max_steps = ocfg.get("max_steps")
        self.max_steps = int(self.max_steps) if self.max_steps is not None else None
        self.use_bf16 = mcfg["use_bfloat16"]

        # ---------- logging ----------
        lcfg: Dict[str, Any] = args.get("logging", {})
        log_dir = Path(lcfg.get("folder", "logs"))
        # log_dir.mkdir(parents=True, exist_ok=True)     
        self.ckpt_dir = log_dir

        self.tag = lcfg.get("write_tag", "train")      
        
        self.csv_logger = CSVLogger(
            str(self.ckpt_dir / f"{self.tag}.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%d", "time (ms)"),
        )

    def _loss_fn(self, h, p) -> torch.Tensor:
        if self.loss_mode == 'l2':
            return F.mse_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        elif self.loss_mode == 'smooth_l1':
            return F.smooth_l1_loss(h.flatten(0,1), p.flatten(0,1), reduction="mean")
        else:
            raise NotImplementedError(f"Loss mode {self.loss_mode} not implemented")

    def _save_ckpt(self, ep, step=None):
        name = f"{self.tag}-step{step}.pth.tar" if step else f"{self.tag}-ep{ep}.pth.tar"
        torch.save({"predictor": self.model.predictor.state_dict(),
                    "backbone_lora": self.model.backbone_lora.state_dict() if self.model.backbone_lora else None,
                    "projector": self.model.projector.state_dict() if self.model.projector else None,
                    "epoch": ep, "lr": self.optimizer.param_groups[0]["lr"]}, self.ckpt_dir/name)

    def train(self):
        mp.set_start_method("spawn", force=True); gstep = 0
        for ep in range(self.epochs):
            logger.info("Epoch %d", ep+1); self.sampler.set_epoch(ep); loss_m, time_m = AverageMeter(), AverageMeter()
            for itr, (imgs, labels, paths) in enumerate(self.loader):
                if self.max_steps is not None and gstep >= self.max_steps:
                    logger.info("Reached max_steps=%d. Stopping training.", self.max_steps)
                    return
                imgs = imgs.to(self.device, non_blocking=True)
                masks_enc, masks_pred = self.mask_generator(batch_size=imgs.size(0), device=self.device)
                def _step():
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        h_raw = self.model.target_features(imgs, paths, n_layer=self.n_layer)
                        h_target = F.layer_norm(h_raw, (h_raw.size(-1),)) if self.target_layer_norm else h_raw
                        h_target = apply_masks(h_target, masks_pred)
                        h_target = repeat_interleave_batch(h_target, imgs.size(0), repeat=len(masks_enc))

                        z = self.model.adapted_features(imgs, paths, n_layer=self.n_layer)
                        z = apply_masks(z, masks_enc)
                        p = self.model.predictor(z, masks_enc, masks_pred)
                        return self._loss_fn(h_target, p)
                (loss,), t = gpu_timer(lambda: [_step()])
                if self.use_bf16: self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                else: loss.backward(); self.optimizer.step()
                grad_stats = grad_logger(
                    list(self.model.predictor.named_parameters())
                    + [(f"backbone_lora.{n}", p) for n, p in self.model.backbone_lora.named_parameters()]
                    if self.model.backbone_lora is not None
                    else self.model.predictor.named_parameters()
                ); self.optimizer.zero_grad()
                loss_m.update(loss.item()); time_m.update(t); gstep += 1
                if gstep % 100 == 0: self._save_ckpt(ep, gstep)
                self.csv_logger.log(ep+1, itr, loss.item(), t)
                if itr % 100 == 0:
                    logger.info("[E %d I %d] loss %.6f (avg %.6f) mem %.2fMB (%.1fms)", ep+1, itr, loss.item(), loss_m.avg, torch.cuda.max_memory_allocated()/1024**2, time_m.avg)
                    if grad_stats:
                        logger.info("    grad: [%.2e %.2e] (%.2e %.2e)", grad_stats.first_layer, grad_stats.last_layer, grad_stats.min, grad_stats.max)
                if self.max_steps is not None and gstep >= self.max_steps:
                    logger.info("Reached max_steps=%d. Stopping training.", self.max_steps)
                    return
            logger.info(
                "Epoch %d complete. Avg loss %.6f, lr %.6f",
                ep + 1,
                loss_m.avg,
                self.optimizer.param_groups[0]['lr']
            )
            if self.scheduler is not None:
                self.scheduler.step()
        if self.max_steps is not None and gstep < self.max_steps:
            logger.warning(
                "Training ended at %d steps before max_steps=%d. Increase optimization.epochs or dataset size.",
                gstep,
                self.max_steps,
            )

def main(args: Dict[str, Any]) -> None:
    if args is None:
        cfg_path = Path(__file__).with_name("params.yaml");
        if not cfg_path.exists(): raise FileNotFoundError("No args provided and default parameter file does not exist")
        with open(cfg_path) as f: args = yaml.safe_load(f)
    Trainer(args).train()

if __name__ == "__main__":
    main()
