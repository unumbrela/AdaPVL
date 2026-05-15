"""
AdaPVL + EVA02-CLIP backbone.

Key differences from medclipseg_evaclip.py:
  1. Uses AdaPVL_Adapter with learnable directional gates (AAGF)
  2. Computes cross-modal alignment scores and regularization loss (CMAS)
  3. Collects intermediate features for multi-layer aggregation (MLFA)
  4. Returns (seg_logits, clip_loss, align_loss) during training
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import open_clip
from .adapvl_layers import AdaPVL_Adapter, MultiLayerAggregator, compute_alignment_loss
from .scale_block import ScaleBlock
from utils.weights import resolve_hf_file


def load_evaclip_to_device(cfg):
    pretrained_path = resolve_hf_file(
        repo_id="timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k",
        filename="open_clip_pytorch_model.bin",
        alt_filenames=["open_clip_model.safetensors"],
        local_subdir="open_clip/eva02_base_patch16_clip_224.merged2b_s8b_b131k",
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        'EVA02-B-16',
        pretrained=str(pretrained_path),
    )
    return model.to(cfg.MODEL.DEVICE).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.trunk = clip_model.visual.trunk
        self.text_model = clip_model.text
        self.logit_scale = clip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        self.embed_dim = 768
        self.text_emb_dim = 512
        self.patch_size = 16
        self.text_proj_dim = 512

        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.dtype = torch.float32

        self.tokenizer = open_clip.get_tokenizer('EVA02-B-16')

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT
        self.use_cmas = cfg.MODEL.get("USE_CMAS", True)
        self.use_mlfa = cfg.MODEL.get("USE_MLFA", True)
        self.share_gates = cfg.MODEL.get("SHARE_GATES", False)
        self.gate_init_vis = cfg.MODEL.get("GATE_INIT_VIS", -3.0)
        self.gate_init_txt = cfg.MODEL.get("GATE_INIT_TXT", 3.0)

        # Segmentation head (unchanged)
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

        # === AAGF: AdaPVL adapters with directional gates ===
        self.pvl_adapters = nn.ModuleList([
            AdaPVL_Adapter(
                in_channels_vis=self.embed_dim,
                in_channels_txt=self.text_emb_dim,
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

        # === MLFA: collect from layers 3, 6, 9 (indices within fusion_stages) ===
        self.mlfa_layers = self.fusion_stages if cfg.MODEL.get("MLFA_ALL_LAYERS", False) else [3, 6, 9]
        self.mlfa = None
        if self.use_mlfa:
            num_collect = len(self.mlfa_layers)
            self.mlfa = MultiLayerAggregator(
                in_dim=self.embed_dim,
                out_dim=self.text_proj_dim,
                num_collect_layers=num_collect,
            )

    def encode_text_image(self, tokenized_prompts, text_prompts, image):
        # === Vision: timm Eva (NLD format) ===
        x_img = self.trunk.patch_embed(image)
        x_img, rot_pos_embed = self.trunk._pos_embed(x_img)
        x_img = self.trunk.norm_pre(x_img)

        # === Text: open_clip TextTransformer (LND format) ===
        x_txt = text_prompts + self.text_model.positional_embedding.type(self.dtype)
        x_txt = x_txt.permute(1, 0, 2)  # NLD -> LND

        intermediate_features = []
        align_scores = []

        for i, (v_block, t_block) in enumerate(zip(
            self.trunk.blocks, self.text_model.transformer.resblocks
        )):
            if i in self.fusion_stages:
                idx = self.fusion_stages.index(i)
                # PVL expects NLD; x_img is NLD, x_txt is LND
                vis_pvl, txt_pvl, align_score = self.pvl_adapters[idx](
                    x_img, x_txt.permute(1, 0, 2)
                )
                # AAGF: gates are applied inside AdaPVL_Adapter
                x_img = x_img + vis_pvl
                x_txt = x_txt + txt_pvl.permute(1, 0, 2)

                # Cache alignment score for CMAS loss computation
                self.pvl_adapters[idx]._cached_align_score = align_score
                align_scores.append(align_score)

            x_img = v_block(x_img, rope=rot_pos_embed)
            x_txt = t_block(x_txt)

            # === MLFA: collect intermediate features ===
            if self.use_mlfa and i in self.mlfa_layers:
                intermediate_features.append(x_img.detach().clone() if not self.training else x_img)

        # === Post-processing ===
        x_img = self.trunk.norm(x_img)
        x_img = self.trunk.head(x_img)  # Linear(768, 512) per-token

        # Text: LND -> NLD, ln_final, select EOS, project
        x_txt = x_txt.permute(1, 0, 2)
        x_txt = self.text_model.ln_final(x_txt)
        x_txt = x_txt[torch.arange(x_txt.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x_txt = x_txt @ self.text_model.text_projection

        # === MLFA: aggregate intermediate features ===
        if self.use_mlfa and len(intermediate_features) > 0:
            x_img = self.mlfa(intermediate_features, x_img)

        return x_img, x_txt

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        seg_feats = image_features[:, 1:, :]  # skip CLS
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
        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.token_embedding(tokenized_prompts).type(self.dtype)

        image_features, text_features = self.encode_text_image(
            tokenized_prompts, prompts, image
        )
        seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)

        if self.training:
            # Contrastive loss (unchanged)
            patch_logits = image_features[:, 1:, :]
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

            # === CMAS: alignment regularization loss ===
            align_loss = compute_alignment_loss(self.pvl_adapters) if self.use_cmas else seg_logits.new_tensor(0.0)

            return seg_logits, clip_loss, align_loss
        else:
            seg_samples = []
            for _ in range(num_samples):
                image_features, text_features = self.encode_text_image(
                    tokenized_prompts, prompts, image
                )
                seg_logits = self.compute_seg_logits(image_features, text_features, B, H, W)
                seg_samples.append(seg_logits)
            return torch.stack(seg_samples, dim=0)


def build_medclipseg_adapvl_evaclip(cfg):
    print("Loading EVA02-CLIP (backbone: EVA02-B-16) with AdaPVL")
    clip_model = load_evaclip_to_device(cfg)
    clip_model.float()

    print("Building AdaPVL + EVA02-CLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if any(k in name for k in ["pvl_adapters", "mask_head", "upscale", "mlfa", "shared_gate_"]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
