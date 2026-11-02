"""
ViT-S/16 backbone for from-scratch continual learning experiments.

Same trimmed-timm VisionTransformer as `backbone/vit.py` (fused-qkv
attention blocks), just the Small configuration (embed_dim 384, 6 heads).
Registered separately so `--backbone vit-s-16 --pretrained 0` reproduces
the ViT-S/16-from-scratch setup used by the consolidation experiments
without touching the default ViT-B registration.
"""

import logging

from backbone import register_backbone
from backbone.vit import create_vision_transformer


@register_backbone("vit-s-16")
def vit_s_16(num_classes: int, pretrained: bool = False):
    if pretrained:
        logging.info("Loading ViT-S/16 with augreg in21k-ft-in1k pre-trained weights.")
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                        num_classes=num_classes)
    variant = 'vit_small_patch16_224.augreg_in21k_ft_in1k' if pretrained \
        else 'vit_small_patch16_224'
    return create_vision_transformer(variant, pretrained=pretrained, **model_kwargs)
