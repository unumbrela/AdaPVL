"""
AdaPVL + DINOv3 + BiomedBERT backbone.

This is the critical test case: the non-aligned encoder pair where
manual one-directional PVL was previously required. With AdaPVL,
the directional gates should automatically learn to close the
vision gate (recovering one-directional behavior) without any
manual configuration.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from transformers import AutoModel, AutoTokenizer
from .adapvl_layers import AdaPVL_Adapter, MultiLayerAggregator, compute_alignment_loss
from .scale_block import ScaleBlock
from utils.weights import resolve_timm_pretrained_overlay, resolve_transformers_source


def load_dinov3_to_device(cfg):
    timm_overlay = resolve_timm_pretrained_overlay('vit_base_patch16_dinov3.lvd1689m')
    biomedbert_dir = resolve_transformers_source(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    vision_model = timm.create_model(
        'vit_base_patch16_dinov3.lvd1689m',
        pretrained=True,
        img_size=cfg.DATASET.SIZE,
        pretrained_cfg_overlay=timm_overlay,
    )
    text_model = AutoModel.from_pretrained(
        biomedbert_dir, local_files_only=True)
    return (vision_model.to(cfg.MODEL.DEVICE).eval(),
            text_model.to(cfg.MODEL.DEVICE).eval())


class CustomCLIP(nn.Module):
    def __init__(self, cfg, vision_model, text_model):
        super().__init__()

        self.cfg = cfg
        self.vision_model = vision_model
        self.text_model = text_model
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        self.embed_dim = 768
        self.patch_size = 16
        self.text_proj_dim = 768
        self.num_prefix_tokens = 5  # 1 CLS + 4 register tokens

        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.dtype = torch.float32

        biomedbert_dir = resolve_transformers_source(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        self.tokenizer = AutoTokenizer.from_pretrained(
            biomedbert_dir, local_files_only=True)

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT
        self.use_cmas = cfg.MODEL.get("USE_CMAS", True)
        self.use_mlfa = cfg.MODEL.get("USE_MLFA", True)
        self.share_gates = cfg.MODEL.get("SHARE_GATES", False)
        self.gate_init_vis = cfg.MODEL.get("GATE_INIT_VIS", -3.0)
        self.gate_init_txt = cfg.MODEL.get("GATE_INIT_TXT", 3.0)

        self.text_proj = nn.Linear(self.embed_dim, self.text_proj_dim)

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.embed_dim) for _ in range(self.num_upscale)],
        )

        # === AAGF: AdaPVL adapters with directional gates ===
        self.pvl_adapters = nn.ModuleList([
            AdaPVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels,
                beta=self.beta,
                gate_init=self.gate_init,
                gate_init_vis=self.gate_init_vis,
                gate_init_txt=self.gate_init_txt,
            )
            for _ in range(len(self.fusion_stages))
        ])

        if self.share_gates and len(self.pvl_adapters) > 0:
            self.shared_gate_vis = nn.Parameter(torch.tensor(self.gate_init_vis))
            self.shared_gate_txt = nn.Parameter(torch.tensor(self.gate_init_txt))
            for adapter in self.pvl_adapters:
                adapter.gate_vis = self.shared_gate_vis
                adapter.gate_txt = self.shared_gate_txt

        # === MLFA: collect from layers 3, 6, 9 ===
        self.mlfa_layers = self.fusion_stages if cfg.MODEL.get("MLFA_ALL_LAYERS", False) else [3, 6, 9]
        self.mlfa = None
        if self.use_mlfa:
            num_collect = len(self.mlfa_layers)
            self.mlfa = MultiLayerAggregator(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                num_collect_layers=num_collect,
            )

    def encode_text_image(self, tokenized_prompts, text_prompts, image,
                          attention_mask=None):
        # === Vision: DINOv3 (NLD, 5 prefix tokens, RoPE) ===
        x_img = self.vision_model.patch_embed(image)
        x_img, rot_pos_embed = self.vision_model._pos_embed(x_img)
        x_img = self.vision_model.norm_pre(x_img)

        # === Text: BiomedBERT (NLD) ===
        x_txt = self.text_model.embeddings(inputs_embeds=text_prompts)

        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.tokenizer.pad_token_id).long()
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask.float()) * torch.finfo(self.dtype).min

        intermediate_features = []

        for i, (v_block, t_layer) in enumerate(zip(
            self.vision_model.blocks,
            self.text_model.encoder.layer
        )):
            if i in self.fusion_stages:
                idx = self.fusion_stages.index(i)
                vis_pvl, txt_pvl, align_score = self.pvl_adapters[idx](
                    x_img, x_txt)
                # AAGF: gates applied inside AdaPVL_Adapter
                # For non-aligned encoders, gate_vis will learn to -> 0
                x_img = x_img + vis_pvl
                x_txt = x_txt + txt_pvl

                self.pvl_adapters[idx]._cached_align_score = align_score

            x_img = v_block(x_img, rope=rot_pos_embed)
            x_txt = t_layer(x_txt, attention_mask=extended_mask)

            if isinstance(x_txt, tuple):
                x_txt = x_txt[0]

            # === MLFA: collect intermediate features ===
            if self.use_mlfa and i in self.mlfa_layers:
                intermediate_features.append(
                    x_img.detach().clone() if not self.training else x_img)

        # === Post-processing ===
        x_img = self.vision_model.norm(x_img)

        # Text: mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        x_txt_pooled = (x_txt * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x_txt = self.text_proj(x_txt_pooled)

        # === MLFA: aggregate ===
        if self.use_mlfa and len(intermediate_features) > 0:
            x_img = self.mlfa(intermediate_features, x_img)

        return x_img, x_txt

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
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
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear",
                                   align_corners=False).squeeze(1)
        return seg_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(self, image, text, num_samples=30):
        B, C, H, W = image.shape

        tokenized_prompts = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=128, return_tensors='pt'
        )['input_ids'].to(self.device)

        with torch.no_grad():
            prompts = self.text_model.embeddings.word_embeddings(tokenized_prompts)

        image_features, text_features = self.encode_text_image(
            tokenized_prompts, prompts, image
        )
        seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)

        if self.training:
            patch_logits = image_features[:, self.num_prefix_tokens:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
            patch_mean = patch_logits.mean(dim=1)

            text_normed = text_features / text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (patch_mean @ text_normed.T) / self.temperature
            logits_per_text = (text_normed @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_normed @ text_normed.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)

            loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
            loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
            clip_loss = (loss_i2t + loss_t2i) / 2

            # === CMAS: alignment regularization ===
            align_loss = compute_alignment_loss(self.pvl_adapters) if self.use_cmas else seg_logits.new_tensor(0.0)

            return seg_logits, clip_loss, align_loss
        else:
            seg_samples = []
            for _ in range(num_samples):
                image_features, text_features = self.encode_text_image(
                    tokenized_prompts, prompts, image
                )
                seg_logits = self.compute_seg_logits(
                    image_features, text_features, B, H, W)
                seg_samples.append(seg_logits)
            return torch.stack(seg_samples, dim=0)


def build_medclipseg_adapvl_dinov3(cfg):
    print("Loading DINOv3 + BiomedBERT with AdaPVL")
    vision_model, text_model = load_dinov3_to_device(cfg)
    vision_model.float()
    text_model.float()

    print("Building AdaPVL + DINOv3")
    model = CustomCLIP(cfg, vision_model, text_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if any(k in name for k in [
            "pvl_adapters", "mask_head", "upscale", "text_proj", "mlfa", "shared_gate_"
        ]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
