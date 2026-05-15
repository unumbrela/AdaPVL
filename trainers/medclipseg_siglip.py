import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import SiglipModel, AutoTokenizer
from .layers import PVL_Adapter
from .scale_block import ScaleBlock
from utils.weights import resolve_transformers_source


def load_siglip_to_device(cfg):
    siglip_dir = resolve_transformers_source("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained(siglip_dir, local_files_only=True)
    return model.to(cfg.MODEL.DEVICE).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, siglip_model):
        super().__init__()

        self.cfg = cfg
        self.vision_model = siglip_model.vision_model
        self.text_model = siglip_model.text_model
        self.logit_scale = siglip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        # SigLIP: both vision and text are 768-dim, no CLS token
        self.embed_dim = 768
        self.patch_size = 16
        self.text_proj_dim = 768  # SigLIP projects to 768

        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE
        self.dtype = torch.float32

        siglip_dir = resolve_transformers_source("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained(siglip_dir, local_files_only=True)

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

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
        # === Vision embeddings (NLD, no CLS token) ===
        x_img = self.vision_model.embeddings(image)  # [B, 196, 768]

        # === Text embeddings (NLD) ===
        x_txt = text_embeds  # already embedded

        # Build text attention mask
        input_ids = tokenized['input_ids']
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # SigLIP uses 4D attention mask
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask.float()) * torch.finfo(self.dtype).min

        for i, (v_layer, t_layer) in enumerate(zip(
            self.vision_model.encoder.layers,
            self.text_model.encoder.layers
        )):
            if i in self.fusion_stages:
                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img, x_txt)
                x_img = x_img + vis_pvl
                x_txt = x_txt + txt_pvl

            x_img = v_layer(x_img, attention_mask=None)
            x_txt = t_layer(x_txt, attention_mask=extended_mask)

            # Handle tuple outputs
            if isinstance(x_img, tuple):
                x_img = x_img[0]
            if isinstance(x_txt, tuple):
                x_txt = x_txt[0]

        # === Post-processing ===
        # Vision: post_layernorm (no CLS to skip, all 196 tokens are patches)
        x_img = self.vision_model.post_layernorm(x_img)

        # Text: final_layer_norm + mean pool
        x_txt = self.text_model.final_layer_norm(x_txt)
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        x_txt_pooled = (x_txt * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return x_img, x_txt_pooled

    def compute_seg_logits(self, image_features, text_features, B, H, W):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # SigLIP has NO CLS token — all tokens are patches
        seg_feats = image_features
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
            # Contrastive loss with all patches (no CLS to skip)
            patch_logits = image_features / image_features.norm(dim=-1, keepdim=True)
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


def build_medclipseg_siglip(cfg):
    print("Loading SigLIP (google/siglip-base-patch16-224)")
    siglip_model = load_siglip_to_device(cfg)
    siglip_model.float()

    print("Building custom SigLIP")
    model = CustomCLIP(cfg, siglip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
