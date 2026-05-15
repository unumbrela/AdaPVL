import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import timm
from transformers import SiglipModel, AutoTokenizer
from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from utils.weights import resolve_timm_pretrained_overlay, resolve_transformers_source


def load_dinov3_siglip_to_device(cfg):
    timm_overlay = resolve_timm_pretrained_overlay('vit_base_patch16_dinov3.lvd1689m')
    siglip_dir = resolve_transformers_source("google/siglip-base-patch16-224")
    vision_model = timm.create_model(
        'vit_base_patch16_dinov3.lvd1689m',
        pretrained=True,
        img_size=cfg.DATASET.SIZE,
        pretrained_cfg_overlay=timm_overlay,
    )
    siglip = SiglipModel.from_pretrained(siglip_dir, local_files_only=True)
    text_model = siglip.text_model
    return vision_model.to(cfg.MODEL.DEVICE).eval(), text_model.to(cfg.MODEL.DEVICE).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, vision_model, text_model):
        super().__init__()

        self.cfg = cfg
        self.vision_model = vision_model  # timm Eva (DINOv3)
        self.text_model = text_model      # SigLIP text encoder
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        self.embed_dim = 768
        self.patch_size = 16
        self.text_proj_dim = 768  # SigLIP text is 768-dim
        self.num_prefix_tokens = 5  # 1 CLS + 4 register

        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.dtype = torch.float32

        siglip_dir = resolve_transformers_source("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained(siglip_dir, local_files_only=True)

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

        # Learnable alignment
        self.vision_proj = nn.Linear(self.embed_dim, self.text_proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels,
                beta=self.beta, gate_init=self.gate_init
            )
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(self, tokenized, text_embeds, image):
        # === Vision: DINOv3 (NLD, 5 prefix tokens, RoPE) ===
        x_img = self.vision_model.patch_embed(image)
        x_img, rot_pos_embed = self.vision_model._pos_embed(x_img)
        x_img = self.vision_model.norm_pre(x_img)

        # === Text: SigLIP text (NLD) ===
        x_txt = text_embeds

        input_ids = tokenized['input_ids']
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask.float()) * torch.finfo(self.dtype).min

        for i, (v_block, t_layer) in enumerate(zip(
            self.vision_model.blocks,
            self.text_model.encoder.layers
        )):
            if i in self.fusion_stages:
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img, x_txt)
                x_img = x_img + vis_pvl
                x_txt = x_txt + txt_pvl

            x_img = v_block(x_img, rope=rot_pos_embed)
            x_txt = t_layer(x_txt, attention_mask=extended_mask)

            if isinstance(x_txt, tuple):
                x_txt = x_txt[0]

        # === Post-processing ===
        x_img = self.vision_model.norm(x_img)
        x_img = self.vision_proj(x_img)  # 768 -> 768 per-token

        # Text: final_layer_norm + mean pool
        x_txt = self.text_model.final_layer_norm(x_txt)
        mask = attention_mask.unsqueeze(-1).float()
        x_txt = (x_txt * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return x_img, x_txt

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Skip prefix tokens (1 CLS + 4 register)
        seg_feats = image_features[:, self.num_prefix_tokens:, :]
        seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)

        h_patch = H // self.patch_size
        w_patch = W // self.patch_size
        seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0, 3, 1, 2)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw",
            self.mask_head(text_features).unsqueeze(1),
            self.upscale(seg_feats)
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)
        return seg_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(self, image, text, num_samples=30):
        B, C, H, W = image.shape

        tokenized = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=64, return_tensors='pt'
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            text_embeds = self.text_model.embeddings(input_ids=tokenized['input_ids'])

        image_features, text_features = self.encode_text_image(tokenized, text_embeds, image)
        seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)

        if self.training:
            patch_logits = image_features[:, self.num_prefix_tokens:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
            patch_mean = patch_logits.mean(dim=1)

            logits_per_image = (patch_mean @ text_features.T) / self.temperature
            logits_per_text = (text_features @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_features @ text_features.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
            clip_loss = (loss_i2t + loss_t2i) / 2
            return seg_logits, clip_loss
        else:
            seg_samples = []
            for _ in range(num_samples):
                image_features, text_features = self.encode_text_image(tokenized, text_embeds, image)
                seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)
                seg_samples.append(seg_logits)
            return torch.stack(seg_samples, dim=0)


def build_medclipseg_dinov3_siglip(cfg):
    print("Loading DINOv3 + SigLIP text")
    vision_model, text_model = load_dinov3_siglip_to_device(cfg)
    vision_model.float()
    text_model.float()

    print("Building custom DINOv3+SigLIP")
    model = CustomCLIP(cfg, vision_model, text_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        elif "vision_proj" in name:
            param.requires_grad_(True)
        elif "logit_scale" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
