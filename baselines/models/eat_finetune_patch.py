"""
Monkey-patch for MaeImageClassificationModel to apply layer_scales[0] to
local_encoder (patch embed) in the d2v_multi path. Upstream EAT skips it,
leaving patch embed at full LR while all transformer blocks use LLRD.

Import this module once before model construction to activate the patch.
"""

import functools


def _patch_mae_image_classification():
    from external.EAT.models.EAT_audio_classification import MaeImageClassificationModel

    original_init = MaeImageClassificationModel.__init__

    @functools.wraps(original_init)
    def patched_init(self, cfg):
        original_init(self, cfg)
        _apply_local_encoder_lr_scale(self, cfg)

    MaeImageClassificationModel.__init__ = patched_init


def _apply_local_encoder_lr_scale(model, cfg):
    if not model.d2v_multi:
        return
    if cfg.layer_decay <= 0:
        return
    if cfg.linear_classifier:
        return

    mod_encs = list(model.model.modality_encoders.values())
    if not mod_encs:
        return

    local_encoder = mod_encs[0].local_encoder
    if local_encoder is None:
        return

    blocks = list(mod_encs[0].context_encoder.blocks) + list(model.model.blocks)
    num_layers = len(blocks) + 1
    # layer_scales[0] is the minimum scale — same id assigned to patch_embed in plain ViT path
    local_encoder_scale = cfg.layer_decay ** num_layers

    for p in local_encoder.parameters():
        optim_override = getattr(p, "optim_overrides", {})
        if "optimizer" not in optim_override:
            optim_override["optimizer"] = {}
        optim_override["optimizer"]["lr_scale"] = local_encoder_scale
        p.optim_overrides = optim_override


_patch_mae_image_classification()
