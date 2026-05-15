from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from open_clip_lib import HFTokenizer, create_model_and_transforms, get_mean_std

from .adapvl_layers import AdaPVL_Adapter, MultiLayerAggregator, compute_alignment_loss
from .scale_block import ScaleBlock
from utils.weights import project_root, resolve_hf_file


def download_checkpoint(filename: str):
    local_path = resolve_hf_file(
        repo_id="TahaKoleilat/MedCLIPSeg",
        filename=f"checkpoints/{filename}",
        local_subdir="checkpoints",
        legacy_local_files=[project_root() / "checkpoints" / filename],
    )
    print(f"Using checkpoint: {local_path}")
    return str(local_path)


def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = "ViT-B-16-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt")
    elif cfg.MODEL.BACKBONE == "ViT-L/14":
        model_name = "ViT-L-14-336-quickgelu"
        pretrained_weights = download_checkpoint("unimed_clip_vit_l14_base_text_encoder.pt")
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    mean, std = get_mean_std()
    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision="amp",
        device=cfg.MODEL.DEVICE,
        force_quick_gelu=True,
        mean=mean,
        std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,
    )
    return model.to(cfg.MODEL.DEVICE).eval()


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
        elif cfg.MODEL.BACKBONE == "ViT-L/14":
            raise NotImplementedError("ViT-L/14 not implemented yet.")
        else:
            raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256,
            **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT
        self.use_cmas = cfg.MODEL.get("USE_CMAS", True)
        self.use_mlfa = cfg.MODEL.get("USE_MLFA", True)
        self.share_gates = cfg.MODEL.get("SHARE_GATES", False)
        self.gate_init_vis = cfg.MODEL.get("GATE_INIT_VIS", -3.0)
        self.gate_init_txt = cfg.MODEL.get("GATE_INIT_TXT", 3.0)

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

        self.mlfa_layers = self.fusion_stages if cfg.MODEL.get("MLFA_ALL_LAYERS", False) else [3, 6, 9]
        self.mlfa = None
        if self.use_mlfa:
            self.mlfa = MultiLayerAggregator(
                in_dim=self.embed_dim,
                out_dim=self.text_proj_dim,
                num_collect_layers=len(self.mlfa_layers),
            )

    def encode_text_image(self, tokenized_prompts, text_prompts, image, attention_mask: Optional[torch.LongTensor] = None):
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        x_txt = self.text_model.transformer.embeddings(inputs_embeds=text_prompts)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)
        x_img = torch.cat(
            [
                self.vision_model.class_embedding.to(x_img.dtype)
                + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
                x_img,
            ],
            dim=1,
        )

        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)

        intermediate_features = []

        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks, self.text_model.transformer.encoder.layer)):
            if i in self.fusion_stages:
                idx = self.fusion_stages.index(i)
                vis_pvl, txt_pvl, align_score = self.pvl_adapters[idx](x_img.transpose(1, 0), x_txt)
                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)
                self.pvl_adapters[idx]._cached_align_score = align_score

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            if isinstance(x_txt, tuple):
                x_txt = x_txt[0]

            if self.use_mlfa and i in self.mlfa_layers:
                feat = x_img.permute(1, 0, 2)
                intermediate_features.append(feat.detach().clone() if not self.training else feat)

        x_img = x_img.permute(1, 0, 2)
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)

        if self.use_mlfa and len(intermediate_features) > 0:
            x_img = self.mlfa(intermediate_features, x_img)

        return x_img, projected

    def compute_seg_logits(self, image_features, text_features, bsz, height, width):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        seg_feats = image_features[:, 1:, :]
        seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)

        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        seg_feats = seg_feats.reshape(bsz, h_patch, w_patch, -1).permute(0, 3, 1, 2)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw",
            self.mask_head(text_features).unsqueeze(1),
            self.upscale(seg_feats),
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)
        return seg_logits

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(self, image, text, num_samples=30):
        bsz, _, height, width = image.shape

        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)

        image_features, text_features = self.encode_text_image(tokenized_prompts, prompts, image)
        seg_logits = self.compute_seg_logits(image_features, text_features, bsz, height, width)

        if self.training:
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
            align_loss = compute_alignment_loss(self.pvl_adapters) if self.use_cmas else seg_logits.new_tensor(0.0)
            return seg_logits, clip_loss, align_loss

        seg_samples = []
        for _ in range(num_samples):
            image_features, text_features = self.encode_text_image(tokenized_prompts, prompts, image)
            seg_logits = self.compute_seg_logits(image_features, text_features, bsz, height, width)
            seg_samples.append(seg_logits)
        return torch.stack(seg_samples, dim=0)


def build_medclipseg_adapvl_unimedclip(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE}) with AdaPVL")
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()

    print("Building AdaPVL + UniMedCLIP")
    model = CustomCLIP(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if any(k in name for k in ["pvl_adapters", "mask_head", "upscale", "mlfa", "shared_gate_"]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
