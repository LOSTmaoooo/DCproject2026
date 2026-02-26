import torch
import torch.nn as nn
from functools import partial
from models.modules._MGViT import MaskedGuidedVisionTransformer

def load_backbone(backbone_name, mask_layers, **kwargs):
    if backbone_name == "dino_vitb16":
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        backbone = MaskedGuidedVisionTransformer(
            patch_size=16, embed_dim=768, depth=12, active_depth=mask_layers, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
        )
    elif backbone_name == "dino_vitb8":
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        backbone = MaskedGuidedVisionTransformer(
            patch_size=8, embed_dim=768, depth=12, active_depth=mask_layers, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
        )
    else:
        raise NotImplementedError(f"Not supported backbone: {backbone_name}")

    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    backbone.load_state_dict(state_dict)
    return backbone