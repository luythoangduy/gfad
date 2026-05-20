
import multiprocessing as mp
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Optional, List
import importlib   
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from src.backbone_lora import build_backbone_lora
from transformers import AutoProcessor, SiglipVisionModel, CLIPVisionModel



class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(
        self,
        model_name: str,
        pred_depth: int,
        pred_emb_dim: int,
        use_cuda: bool = True,
        if_pe: bool = True,
        feat_normed: bool = False,
        gated_attention: Optional[Dict[str, Any]] = None,
        backbone_gating: Optional[Dict[str, Any]] = None,
        backbone_lora: Optional[Dict[str, Any]] = None,
        weights: Optional[str] = None,
        crop_size: Optional[int] = None,
    ):
        super().__init__()
        self.weights = weights
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.num_patches = self._num_patches_for_crop(model_name, crop_size, self.num_patches)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](num_patches=self.num_patches, embed_dim=self.embed_dim,
                                                         predictor_embed_dim=pred_emb_dim, depth=pred_depth, if_pe=if_pe, feat_normed=feat_normed)
        self._init_predictor(self.predictor)
        if gated_attention and gated_attention.get("enabled", False):
            print("Predictor gated_attention is ignored on this branch; use meta.backbone_lora instead.")
        if backbone_lora is None:
            backbone_lora = backbone_gating
        self.backbone_lora = build_backbone_lora(self.encoder, self.embed_dim, backbone_lora)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()
        self.feat_normed = self.predictor.feat_normed # it depends on the predictor
        print(f"Normed features: {self.feat_normed}")

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)
    
    def target_features(self, images, paths, n_layer=3):
        lora_context = self.backbone_lora.use_adapters(False) if self.backbone_lora is not None else nullcontext()
        with lora_context, torch.no_grad():
            return self._extract(images, paths, n_layer=n_layer)

    def adapted_features(self, images, paths, n_layer=3):
        lora_context = self.backbone_lora.use_adapters(True) if self.backbone_lora is not None else nullcontext()
        with lora_context:
            return self._extract(images, paths, n_layer=n_layer)

    def gated_features(self, images, paths, n_layer=3):
        return self.adapted_features(images, paths, n_layer=n_layer)

    def context_features(self, images, paths, n_layer=3):
        z = self.adapted_features(images, paths, n_layer=n_layer)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):

        projector = processor = None
        if model == "dinov2":
            enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval(); num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3":
            if not self.weights:
                raise ValueError("meta.weights is required when meta.model is 'dinov3'")
            enc = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", source="github", weights=self.weights).eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval(); num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval(); processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512"); num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval(); processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16"); num_patches, embed_dim = 196, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config      
            
            config = Config()
            enc = init_vit_backbone(config)

            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")
        if model != 'dinosiglip':
            for p in enc.parameters(): 
                p.requires_grad = False
        if projector is not None:
            for p in projector.parameters():
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    @staticmethod
    def _num_patches_for_crop(model: str, crop_size: Optional[int], fallback: int) -> int:
        if crop_size is None:
            return fallback
        patch_sizes = {
            "dinov2": 14,
            "dinov3": 16,
            "dino": 16,
            "siglip": 16,
            "clip": 16,
        }
        patch_size = patch_sizes.get(model)
        if patch_size is None:
            return fallback
        return (int(crop_size) // patch_size) ** 2

    def _extract(self, imgs: torch.Tensor, paths: List[str], n_layer: int = 3):
        if self.model_name == "dinov2":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] # the thrid last block
        elif self.model_name == "dinov3":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer, return_class_token=False)[0] 
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=n_layer)[0][:,1:,:]
        elif self.model_name == "siglip":
            pil_list = [Image.open(p).convert("RGB") for p in paths]
            proc = self.processor(images=pil_list, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(imgs.device)

            with torch.no_grad():
                out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
                hs = out.hidden_states  # tuple: [embeddings, block1, ..., blockL]; len = L+1

            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, :, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "clip":
            hs = self.encoder(pixel_values=imgs, output_hidden_states=True).hidden_states
            L = len(hs) - 1  # number of transformer blocks
            n = max(1, min(n_layer, L))
            h = hs[-n][:, 1:, :]   # [B, 1024, 768] for 512/16 patches
            # print(h.shape)
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0,2,1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)

        if self.feat_normed:
            h = F.normalize(h, dim=-1)

        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
