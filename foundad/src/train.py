
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
from src.utils.synthesis import CutPasteUnion
from src.foundad import VisionModule
from src.gated_attention_projector import gate_supervision_loss

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
            weights=mcfg.get("weights"),
        )
        self.n_layer = args["meta"].get("n_layer", 3)
        self.model.predictor.requires_grad_(True)
        if self.model.projector:
            self.model.projector.requires_grad_(True)
        self.loss_mode = args["meta"].get("loss_mode", "l2") # l2 or smooth_l1
        logger.info(f"Loss mode {self.loss_mode}")
        gate_supervision_cfg = mcfg.get("gated_attention", {}).get("supervision", {})
        self.gate_supervision_enabled = bool(gate_supervision_cfg.get("enabled", False))
        self.gate_supervision_weight = float(gate_supervision_cfg.get("loss_weight", 0.1))
        self.gate_supervision_anomaly_weight = float(gate_supervision_cfg.get("anomaly_weight", 4.0))
        self.gate_supervision_positions = gate_supervision_cfg.get("positions")

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
        self.cutpaste = CutPasteUnion(colorJitter=0.5)
        self.batch_size = dcfg["batch_size"]

        # ---------- optimization ----------
        from src.helper import init_opt

        ocfg = args["optimization"]
        self.optimizer, self.scheduler, self.scaler = init_opt(
            predictor=self.model.predictor,
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
                _, imgs_abn, anomaly_masks = self.cutpaste(imgs, labels) # anomaly synthesis on CPU
                imgs = imgs.to(self.device, non_blocking=True)
                imgs_abn = imgs_abn.to(self.device, non_blocking=True)
                anomaly_masks = anomaly_masks.to(self.device, non_blocking=True)
                def _step():
                    with autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
                        if np.random.rand() < 0.5:
                            h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs, paths, n_layer=self.n_layer)
                            gate_labels = torch.zeros_like(anomaly_masks)
                        else:
                            h = self.model.target_features(imgs, paths, n_layer=self.n_layer); _, p = self.model.context_features(imgs_abn, paths, n_layer=self.n_layer)
                            gate_labels = anomaly_masks
                        loss = self._loss_fn(h, p,)
                        if self.gate_supervision_enabled:
                            gate_loss = gate_supervision_loss(
                                predictor=self.model.predictor,
                                anomaly_mask=gate_labels,
                                num_tokens=p.shape[1],
                                positions=self.gate_supervision_positions,
                                anomaly_weight=self.gate_supervision_anomaly_weight,
                            )
                            if gate_loss is not None:
                                loss = loss + self.gate_supervision_weight * gate_loss
                        return loss
                (loss,), t = gpu_timer(lambda: [_step()])
                if self.use_bf16: self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
                else: loss.backward(); self.optimizer.step()
                grad_stats = grad_logger(self.model.predictor.named_parameters()); self.optimizer.zero_grad()
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
