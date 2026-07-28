# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Callable, List
from functools import partial
from omegaconf import II
from enum import Enum, auto
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from external.EAT.models.base import (
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder, 
    get_annealed_rate,
)

from external.EAT.models.modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from external.EAT.models.images import (
    D2vImageConfig,
    ImageEncoder,
)

from external.dinov2.dinov2.layers.dino_head import DINOHead
from external.dinov2.dinov2.loss.dino_clstoken_loss import DINOLoss

logger = logging.getLogger(__name__)

# we follow the work of data2vec 2.0 on image modality and Audio-MAE in EAT
class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


# ---------------------------------------------------------------------------
# Emotion-correlated prosody target (see emotion-correlated-prosody.md)
#
# Three analytic per-frame descriptors -- log-energy, spectral centroid,
# spectral flux -- pooled to the patch grid's time resolution and predicted at
# masked positions. Label-free: computed from the spectrogram the model already
# ingests.
# ---------------------------------------------------------------------------


def prosody_valid_time(source, pad_value, n_time_patches, patch_frames, tol=1e-2):
    """Route B pad detection: which time-patches are entirely non-padding.

    Padding is zero-padded in raw log-mel *before* the dataset's global
    normalization (raw_audio_dataset.py:396-410), so padded frames sit at exactly
    `pad_value` across all 128 mel bins -- a flatness real speech never produces.

    Tested on the *normalized* `source` with a tolerance, deliberately:
      - training runs in bf16 (common.bf16), whose ulp near pad_value is ~2e-3,
        so an exact comparison never fires and every frame reads as valid;
      - de-normalizing first would multiply that error by ~9.14.
    tol=1e-2 is ~5 bf16 ulps here, and corresponds to all 128 bins agreeing to
    within ~0.09 in log-mel -- far flatter than any real frame.

    A time-patch is valid only if it lies entirely within the valid region, so the
    patch straddling n_frames is discarded (at most 160ms).
    """
    pad_frame = ((source - pad_value).abs() < tol).all(dim=-1)  # (B, T)
    # mark the *trailing* run of pad frames only: cumprod from the end stays 1
    # while every frame seen so far (scanning backwards) is padding.
    trailing_pad = torch.cumprod(pad_frame.flip(-1).long(), dim=-1).flip(-1)
    n_frames = (1 - trailing_pad).sum(-1)  # (B,)
    n_valid = torch.div(n_frames, patch_frames, rounding_mode="floor")

    idx = torch.arange(n_time_patches, device=source.device)
    return idx.unsqueeze(0) < n_valid.unsqueeze(1)  # (B, n_time_patches)


def compute_prosody(S_log, valid_time, n_time_patches, patch_frames, norm, corpus_mean=None, corpus_std=None):
    """Analytic prosody descriptors per time-patch.

    S_log:      (B, T, M) raw natural-log mel -- NOT the globally normalized `source`.
    valid_time: (B, n_time_patches) bool, from prosody_valid_time (Route B) or a
                threaded n_frames (Route A). Kept as a parameter so the two routes
                differ in one call site only.
    Returns prosody, shape (B, 3, n_time_patches).
    """
    B, T, M = S_log.shape

    log_energy = torch.logsumexp(S_log, dim=-1)  # (B, T)

    bins = torch.arange(M, device=S_log.device, dtype=S_log.dtype)
    centroid = (torch.softmax(S_log, dim=-1) * bins).sum(-1)  # (B, T)

    # S_{-1} := S_0, so flux[0] == 0. Chosen explicitly: dropping frame 0 instead
    # would misalign every patch boundary by one frame.
    delta = S_log[:, 1:] - S_log[:, :-1]
    flux = torch.cat(
        [S_log.new_zeros(B, 1), torch.linalg.vector_norm(delta, dim=-1)], dim=1
    )  # (B, T)

    p = torch.stack([log_energy, centroid, flux], dim=1)  # (B, 3, T)
    p = p.view(B, 3, n_time_patches, patch_frames).mean(-1)  # (B, 3, n_time_patches)

    vt = valid_time.unsqueeze(1).to(p.dtype)  # (B, 1, n_time_patches)
    if norm == "instance":
        # per-utterance, per-descriptor over time -- valid patches only, so the
        # pad plateau cannot pull the statistics.
        cnt = vt.sum(-1, keepdim=True).clamp(min=1.0)
        mean = (p * vt).sum(-1, keepdim=True) / cnt
        var = (((p - mean) ** 2) * vt).sum(-1, keepdim=True) / cnt
        p = (p - mean) / (var + 1e-6).sqrt()
    elif norm == "corpus":
        assert corpus_mean is not None and corpus_std is not None, (
            "prosody_norm=corpus requires prosody_corpus_mean/std; compute them "
            "offline over the pretraining manifest using this same function"
        )
        mean = torch.as_tensor(corpus_mean, device=p.device, dtype=p.dtype).view(1, 3, 1)
        std = torch.as_tensor(corpus_std, device=p.device, dtype=p.dtype).view(1, 3, 1)
        p = (p - mean) / (std + 1e-6)
    elif norm == "none":
        # Raw descriptors, in their natural units. Diagnostic-only: used by the
        # pre-flight probe, where absolute level is the thing under test, and by
        # the offline corpus-statistics job, which must see unnormalized values to
        # produce the constants `corpus` mode consumes. Never set in a training
        # config -- an unnormalized target would make PROSODY_DIM_PARITY (which
        # assumes ~unit target variance) meaningless.
        pass
    else:
        raise ValueError(f"unknown prosody_norm: {norm}")

    return p * vt


# --- probe-only descriptor candidates -------------------------------------
# NOT used in training. compute_prosody above is the training target and is
# untouched; these exist so Pre-flight A can report whether a candidate earns a
# place in it, before a run is spent and before the target becomes expensive to
# change. Promoting one is a deliberate edit to compute_prosody, not a flag.

PROSODY_CANDIDATES = ["alpha_ratio", "hammarberg"]


def mel_band_indices(n_mels, sample_rate, f_lo, f_hi, low_freq=20.0, high_freq=None):
    """Mel-bin indices whose centre frequency falls in [f_lo, f_hi).

    Derived from n_mels / sample_rate rather than hardcoded: the band edges move
    if either changes, and a stale constant would silently select the wrong part
    of the spectrum. Defaults match torchaudio's kaldi fbank (low_freq=20,
    high_freq=Nyquist), which is what raw_audio_dataset.py:392 calls.
    """
    high_freq = high_freq if high_freq else sample_rate / 2.0
    to_mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)
    edges = np.linspace(to_mel(low_freq), to_mel(high_freq), n_mels + 2)
    centers = edges[1:-1]
    sel = np.where((centers >= to_mel(f_lo)) & (centers < to_mel(f_hi)))[0]
    assert len(sel), f"no mel bins in [{f_lo}, {f_hi}) Hz at n_mels={n_mels}, sr={sample_rate}"
    return int(sel[0]), int(sel[-1]) + 1


def compute_prosody_candidates(S_log, valid_time, n_time_patches, patch_frames, sample_rate=16000):
    """Raw (unnormalized) candidate descriptors, for the pre-flight probe only.

    Both are eGeMAPS *minimalistic-set* spectral-balance parameters, and both are
    exactly computable from log-mel as DIFFERENCES of log-domain quantities --
    which makes them exactly gain-invariant, unlike their linear-magnitude
    definitions:

      alpha ratio  log energy 50-1000 Hz  minus  log energy 1-5 kHz
      hammarberg   log peak   0-2000 Hz   minus  log peak   2-5 kHz

    Returns (B, 2, n_time_patches), zeroed at invalid patches, matching
    compute_prosody's convention so the two can be concatenated.
    """
    B, T, M = S_log.shape
    a_lo = mel_band_indices(M, sample_rate, 50.0, 1000.0)
    a_hi = mel_band_indices(M, sample_rate, 1000.0, 5000.0)
    h_lo = mel_band_indices(M, sample_rate, 0.0, 2000.0)
    h_hi = mel_band_indices(M, sample_rate, 2000.0, 5000.0)

    alpha = (torch.logsumexp(S_log[..., a_lo[0]:a_lo[1]], dim=-1)
             - torch.logsumexp(S_log[..., a_hi[0]:a_hi[1]], dim=-1))
    hamm = (S_log[..., h_lo[0]:h_lo[1]].amax(-1)
            - S_log[..., h_hi[0]:h_hi[1]].amax(-1))

    p = torch.stack([alpha, hamm], dim=1)                       # (B, 2, T)
    p = p.view(B, 2, n_time_patches, patch_frames).mean(-1)
    return p * valid_time.unsqueeze(1).to(p.dtype)


def prosody_interp_baseline(target, anchor):
    """Linear interpolation of `target` from the nearest anchor columns.

    target: (B, n_time, 3) -- the prosody target on the full time grid.
    anchor: (B, n_time) bool -- columns the model can actually see.

    Rows with no anchor at all are NOT handled here: they would yield an arbitrary
    constant rather than an interpolation. The caller must exclude them (the loss
    site does, via `col_t & anchor.any(-1, keepdim=True)`), so that the loss and
    this baseline are scored on identical positions.

    The triviality diagnostic: if the model cannot beat this, the prosody task is
    solvable by interpolation and will not reshape the encoder at any lambda.
    """
    B, n_time, _ = target.shape
    idx = torch.arange(n_time, device=target.device).unsqueeze(0).expand(B, -1)

    left = torch.cummax(torch.where(anchor, idx, torch.full_like(idx, -1)), dim=1).values
    right = torch.cummin(
        torch.where(anchor, idx, torch.full_like(idx, n_time)).flip(1), dim=1
    ).values.flip(1)

    # columns with an anchor on only one side fall back to that side; the
    # clamps make the gather safe and the weight below collapses to it.
    has_left, has_right = left >= 0, right < n_time
    l = left.clamp(min=0)
    r = right.clamp(max=n_time - 1)
    l = torch.where(has_left, l, r)
    r = torch.where(has_right, r, l)

    span = (r - l).clamp(min=1).to(target.dtype)
    w = ((idx - l).to(target.dtype) / span).unsqueeze(-1)  # (B, n_time, 1)

    y_l = torch.gather(target, 1, l.unsqueeze(-1).expand(-1, -1, 3))
    y_r = torch.gather(target, 1, r.unsqueeze(-1).expand(-1, -1, 3))
    return y_l * (1 - w) + y_r * w


def _r2(pred, target):
    """Pooled R^2 over the selected positions, averaged across descriptors."""
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0, keepdim=True)) ** 2).sum(0)
    return (1 - ss_res / (ss_tot + 1e-8)).mean()

@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    image: D2vImageConfig = D2vImageConfig()
    
@dataclass
class Data2VecMultiConfig(FairseqDataclass):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 12
    
    # standard vision Transformer
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False
    layer_decay: float = 1.0
    no_decay_blocks: bool = True

    # EAT averages all Transformer block output (12 layers in total) 
    average_top_k_layers: int = field(
        default=12, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    # clone batch for multi-mask strategy
    clone_batch: int = 16

    # normalization for teacher Transformer layer output
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    # EMA settings
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    ema_anneal_end_step: int = II("optimization.max_update")

    # In EAT, the Transformer encoder and the CNN encoder are both EMA updated
    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")

    skip_ema: bool = False

    # d2v_loss is the frame-level loss while cls_loss is the utterance-level loss
    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    # emotion-correlated prosody target (see emotion-correlated-prosody.md).
    # 0 disables it entirely; the existing path is untouched when off.
    prosody_loss: float = field(
        default=0,
        metadata={"help": "weight for the analytic prosody target; 0 disables. 1.0 is parity with recon=1 (the dim-parity factor is applied internally)"},
    )
    prosody_norm: str = field(
        default="instance",
        metadata={"help": "'instance' (per-utterance over time, contour only) or 'corpus' (fixed stats, preserves absolute level). lambda does NOT transfer between modes"},
    )
    prosody_corpus_mean: Optional[List[float]] = field(
        default=None, metadata={"help": "3 per-descriptor means for prosody_norm=corpus"}
    )
    prosody_corpus_std: Optional[List[float]] = field(
        default=None, metadata={"help": "3 per-descriptor stds for prosody_norm=corpus"}
    )
    # must match the dataset's global normalization (raw_audio_dataset.py:407-408);
    # the descriptors are nonlinear in S, so they are wrong on normalized input.
    prosody_denorm_mean: float = -4.268
    prosody_denorm_std: float = 4.569
    prosody_diag_interval: int = field(
        default=1000,
        metadata={"help": "steps between triviality-diagnostic computations; match common.log_interval"},
    )

    decoder_group: bool = False

    softmax_temperature_student: float = field(default=0.1, metadata={"help": "student temperature for DINOLoss"})

    use_dino_head: bool = field(default=False, metadata={"help": "if true, use DINOHead MLP + DINOLoss for CLS loss; if false, use MSE against mean-pooled teacher patches"})
    dino_out_dim: int = field(default=65536, metadata={"help": "DINO prototype dimension"})
    dino_nlayers: int = field(default=3, metadata={"help": "DINOHead MLP layers"})
    dino_hidden_dim: int = field(default=2048, metadata={"help": "DINOHead hidden dim"})
    dino_bottleneck_dim: int = field(default=256, metadata={"help": "DINOHead bottleneck dim"})
    dino_teacher_temp: float = field(default=0.04, metadata={"help": "teacher sharpening temperature for DINOLoss"})
    dino_teacher_temp_warmup_init: float = field(default=0.04, metadata={"help": "initial teacher temp for warmup (set higher than dino_teacher_temp to enable warmup)"})
    dino_teacher_temp_warmup_steps: int = field(default=0, metadata={"help": "number of steps to linearly anneal teacher temp from warmup_init to dino_teacher_temp; 0 disables warmup"})
    dino_center_momentum: float = field(default=0.9, metadata={"help": "EMA momentum for DINOLoss center buffer"})


@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel):
    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> ModalitySpecificEncoder:
        if cfg.type.value == Modality.IMAGE.value:
            enc_cls = ImageEncoder
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        
        # extract CNN encoder and CNN decoder from modified data2vec image modality (see image.py)
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            enc = self.make_modality_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )
            self.modality_encoders[mod.name] = enc

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        if self.cfg.mae_init:
            self.apply(self._init_weights)
        else:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()

        # make teacher model
        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay)
            self.shared_decoder = (
                Decoder1d(cfg.shared_decoder, cfg.embed_dim)
                if self.cfg.shared_decoder is not None
                else None
            )
            if self.shared_decoder is not None:
                self.shared_decoder.apply(self._init_weights)

            self.recon_proj = None
            if cfg.recon_loss > 0:
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim//3)

            self.prosody_proj = None
            if cfg.prosody_loss > 0:
                self.prosody_proj = nn.Linear(cfg.embed_dim, 3)
                # Route B recovers padding as the trailing run of exactly-constant
                # frames. Augmentation destroys exact constancy and torch.roll
                # circularly shifts padding off the tail, corrupting the targets
                # with no error -- fail loudly instead.
                assert not getattr(getattr(task, "cfg", None), "noise", False), (
                    "prosody_loss requires task.noise=False (Route B pad detection); "
                    "thread n_frames through the dataset (Route A) to lift this"
                )
                # Validate here rather than inside compute_prosody, which first
                # runs one training step in -- after model build, EMA teacher
                # construction and dataloader warm-up. A typo would otherwise cost
                # minutes of a job to surface. "none" is deliberately excluded: it
                # is a diagnostic mode for the probes and the corpus-stats job, and
                # an unnormalized target makes the dim-parity constant (which
                # assumes ~unit target variance) meaningless.
                assert cfg.prosody_norm in ("instance", "corpus"), (
                    f"prosody_norm={cfg.prosody_norm!r}; training accepts only "
                    "'instance' or 'corpus' ('none' is diagnostic-only)"
                )
                # PROSODY_DIM_PARITY is computed from recon's feature dim
                # (patch_size^2 * in_chans = 256), so "prosody_loss=1.0 is parity"
                # is a statement about `recon` specifically. With recon disabled
                # the constant still applies but the contract it documents is
                # false: d2v's feature dim is embed_dim=768, not 256, so a lambda
                # sweep would be calibrated against a term that is not running.
                # Note the dataclass defaults are recon_loss=0 / d2v_loss=1 -- the
                # broken combination -- and only the shipped yaml flips them.
                assert cfg.recon_loss > 0, (
                    f"prosody_loss>0 requires recon_loss>0 (got {cfg.recon_loss}); "
                    "the dim-parity constant that makes prosody_loss=1.0 mean "
                    "'parity with recon=1' is derived from recon's 256-dim target. "
                    "Prosody is designed to sit alongside recon, not replace it."
                )
                if cfg.prosody_norm == "corpus":
                    assert (
                        cfg.prosody_corpus_mean is not None
                        and cfg.prosody_corpus_std is not None
                        and len(cfg.prosody_corpus_mean) == 3
                        and len(cfg.prosody_corpus_std) == 3
                    ), (
                        "prosody_norm=corpus needs 3 prosody_corpus_mean and 3 "
                        "prosody_corpus_std; produce them with "
                        "scripts/compute_prosody_corpus_stats.py"
                    )
                    # roll_mag_aug multiplies the waveform by a random gain
                    # mag ~ Beta(10,10)+0.5 in [0.5, 1.5]
                    # (raw_audio_dataset.py:87-92), which shifts every log-mel bin
                    # by 2*ln(mag) and so log_energy by -1.39..+0.81 nats, redrawn
                    # every epoch. Centroid and flux are invariant to it -- a
                    # uniform level shift cancels in softmax(S) and in S_t - S_t-1
                    # -- so this bites exactly one descriptor, and only in `corpus`
                    # mode: `instance` subtracts the per-utterance mean and removes
                    # the offset entirely. In corpus mode the absolute level the
                    # mode exists to preserve becomes augmentation noise,
                    # mis-centered against constants that
                    # scripts/compute_prosody_corpus_stats.py produced with the
                    # augmentation off. Nothing in the loss curve reveals it.
                    assert not getattr(getattr(task, "cfg", None), "roll_aug", False), (
                        "prosody_norm=corpus requires task.roll_aug=False; the random "
                        "gain randomizes absolute log-energy, which is the only thing "
                        "corpus mode adds over instance. Use prosody_norm=instance to "
                        "keep roll_aug."
                    )

            self.student_dino_head = None
            self.teacher_dino_head = None
            self.dino_loss_fn = None
            if cfg.use_dino_head:
                self.student_dino_head = DINOHead(
                    cfg.embed_dim, cfg.dino_out_dim,
                    nlayers=cfg.dino_nlayers, hidden_dim=cfg.dino_hidden_dim,
                    bottleneck_dim=cfg.dino_bottleneck_dim,
                )
                self.teacher_dino_head = DINOHead(
                    cfg.embed_dim, cfg.dino_out_dim,
                    nlayers=cfg.dino_nlayers, hidden_dim=cfg.dino_hidden_dim,
                    bottleneck_dim=cfg.dino_bottleneck_dim,
                )
                for p_s, p_t in zip(self.student_dino_head.parameters(), self.teacher_dino_head.parameters()):
                    p_t.data.copy_(p_s.data)
                self.teacher_dino_head.requires_grad_(False)
                self.dino_loss_fn = DINOLoss(
                    cfg.dino_out_dim,
                    student_temp=cfg.softmax_temperature_student,
                    center_momentum=cfg.dino_center_momentum,
                )

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                optim_override = getattr(p, "optim_overrides", {})
                optimizer_cfg = optim_override.get("optimizer", {})
                optimizer_cfg["weight_decay_scale"] = 0
                optim_override["optimizer"] = optimizer_cfg
                p.optim_overrides = optim_override
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        if cfg.layer_decay > 0:
            blocks = []
            for mod_enc in self.modality_encoders.values():
                context_blocks = getattr(mod_enc.context_encoder, "blocks", None)
                if context_blocks is not None:
                    blocks.extend(list(context_blocks))
            blocks.extend(list(self.blocks))

            if blocks:
                num_layers = len(blocks) + 1
                layer_scales = [
                    cfg.layer_decay ** (num_layers - i)
                    for i in range(num_layers + 1)
                ]

                for i, block in enumerate(blocks):
                    lid = i + 1
                    lr_scale = layer_scales[lid]
                    if lr_scale == 1.0:
                        continue
                    for _, p in block.named_parameters():
                        optim_override = getattr(p, "optim_overrides", {})
                        if "optimizer" not in optim_override:
                            optim_override["optimizer"] = {}
                        if cfg.no_decay_blocks:
                            optim_override["optimizer"]["lr_scale"] = lr_scale
                        else:
                            optim_override["optimizer"] = {"lr_scale": lr_scale}
                        p.optim_overrides = optim_override
        
        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def make_ema_teacher(self, ema_decay):
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,
            ema_fp32=True,
            log_norms=self.cfg.log_norms,
            add_missing_params=False,
        )

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    # teacher model (with independent CNN encoder and Transformer encoder)
    def make_target_model(self):
        logger.info("making target model")

        model_copy = Data2VecMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=None
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    # teacher model updated with EMA
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)
                if self.student_dino_head is not None:
                    decay = self.ema.get_decay()
                    for p_s, p_t in zip(self.student_dino_head.parameters(), self.teacher_dino_head.parameters()):
                        p_t.data.mul_(decay).add_(p_s.data, alpha=1 - decay)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]
            )
        else:
            modalities = task.supported_modalities
        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        feature_extractor = self.modality_encoders[mode]

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        # extract (unmasked) features using CNN encoder
        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        # x in shape ( batch_size * clone batch, patch_frame(64) * patch_freqency(8) * unmask_ratio(0.2) + 1(cls_token), 768(feature dimension) )
        # EAT does not employ the ablibi mechanism in Transformer
        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        # standard Transformer (for student encoder)
        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # extract features for fine-tuning
        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        # decode features merged with masked tokens, dx in shape (batch_size * clone_batch, patch, 768)
        xs = []
        if self.shared_decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                self.shared_decoder,
                encoder_mask,
            )
            xs.append(dx)
        if feature_extractor.decoder is not None:
            dx = self.forward_decoder(
                x,
                feature_extractor,
                feature_extractor.decoder,
                encoder_mask,
            )
            xs.append(dx)
            orig_x = x

        assert len(xs) > 0

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        tm = self.ema.model

        # encode audio spectrogram using teacher model
        with torch.no_grad():
            tm.eval()

            if self.cfg.ema_encoder_only:
                assert target is None
                ema_input = extractor_out["local_features"]
                ema_input = feature_extractor.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                if feature_extractor.modality_cfg.ema_local_encoder:
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    ema_input = tm.modality_encoders[mode](
                        inp,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )
                else:
                    assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_feature_enc = tm.modality_encoders[mode]
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input.get("alibi_bias", None)
            ema_alibi_scale = ema_input.get("alibi_scale", None)
            ema_input = ema_input["x"]

            # extract target features using teacher CNN encoder
            # ema_input in shape (batch_size, patch + 1(cls_token), feature_dimension)
            y = []
            ema_x = []
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):  
                ab = ema_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                ema_input, lr = blk(
                    ema_input,
                    padding_mask=ema_padding_mask,
                    alibi_bias=ab,
                )
                y.append(lr[:, extra_tokens:])
                ema_x.append(ema_input[:, extra_tokens:])

            teacher_cls_logits = None
            if self.teacher_dino_head is not None:
                teacher_cls = ema_input[:, 0].float()  # (B, embed_dim)
                self.teacher_dino_head.float()  # weight_norm kernel requires float32
                teacher_cls_logits = self.teacher_dino_head(teacher_cls)  # (B, dino_out_dim)

        # EAT utilize total 12 Transformer block layer output average as target
        y = self.make_targets(y, self.average_top_k_layers)
        orig_targets = y

        # multiply the target value according to the number of clone batch
        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        # extract values in masked position to make prediction
        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        y = y[masked_b]     

        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
        else:
            xs = [x.reshape(-1, x.size(-1)) for x in xs]
            

        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        # MSE CLS loss: student CLS predicts mean-pooled teacher patch targets
        if self.cfg.cls_loss > 0 and not self.cfg.use_dino_head:
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)
            cls_pred = x[:, extra_tokens - 1]

            result["losses"]["cls"] = self.d2v_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )

        # DINO loss with DINOHead MLP and proper prototype space
        if self.cfg.cls_loss > 0 and self.cfg.use_dino_head:
            assert extra_tokens > 0
            assert teacher_cls_logits is not None

            student_cls = x[:, extra_tokens - 1].float()            # (B*clone, embed_dim)
            self.student_dino_head.float()  # weight_norm kernel requires float32
            student_logits = self.student_dino_head(student_cls)    # (B*clone, dino_out_dim)

            if self.cfg.clone_batch > 1:
                teacher_logits_rep = teacher_cls_logits.repeat_interleave(self.cfg.clone_batch, 0)
            else:
                teacher_logits_rep = teacher_cls_logits

            if self.cfg.dino_teacher_temp_warmup_steps > 0 and self.num_updates < self.cfg.dino_teacher_temp_warmup_steps:
                warmup_progress = self.num_updates / self.cfg.dino_teacher_temp_warmup_steps
                teacher_temp = self.cfg.dino_teacher_temp_warmup_init + warmup_progress * (self.cfg.dino_teacher_temp - self.cfg.dino_teacher_temp_warmup_init)
            else:
                teacher_temp = self.cfg.dino_teacher_temp

            teacher_soft = self.dino_loss_fn.softmax_center_teacher(
                teacher_logits_rep, teacher_temp
            )
            self.dino_loss_fn.update_center(teacher_cls_logits)  # B samples (not repeated)

            result["losses"]["cls"] = self.dino_loss_fn(
                [student_logits], [teacher_soft]
            ) * (self.cfg.cls_loss * sample_size)

            with torch.no_grad():
                result["dino_center_norm"] = self.dino_loss_fn.center.norm()
                result["dino_center_std"] = self.dino_loss_fn.center.std()
                result["dino_teacher_entropy"] = -(teacher_soft * torch.log(teacher_soft + 1e-8)).sum(dim=-1).mean()
                result["dino_active_prototypes"] = (teacher_soft.max(dim=0).values > 1e-4).sum().float()

        if self.cfg.recon_loss > 0:

            with torch.no_grad():
                target = feature_extractor.patchify(source)  #(btz,1,512,16*16)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5   #(btz,1,512,1)

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(self.cfg.clone_batch, 0)  #(btz*clone_btz,1,512,1)

                if masked_b is not None:
                    target = target[masked_b]

            recon = xs[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss * sample_size
            )

        if self.cfg.prosody_loss > 0:

            with torch.no_grad():
                n_time, n_freq = feature_extractor.hw          # (64, 8)
                patch_frames = feature_extractor.modality_cfg.patch_size

                src = source.squeeze(1)                            # (B, T, M)
                denorm_scale = self.cfg.prosody_denorm_std * 2

                # Pad detection runs on the NORMALIZED input with a tolerance:
                # training is bf16, so an exact test never fires, and denormalizing
                # first would amplify the error ~9.14x. See prosody_valid_time.
                valid_time = prosody_valid_time(
                    src, -self.cfg.prosody_denorm_mean / denorm_scale, n_time, patch_frames
                )                                                  # (B, 64)

                # Step 0: recover raw log-mel. The descriptors are nonlinear in S,
                # so computing them on the normalized `source` yields a
                # near-degenerate centroid -- silently, with no error.
                # .float(): the batch arrives in bf16 (common.bf16) and the
                # instance-norm variance below is not safe at that precision. This
                # cannot recover bits already lost -- the descriptors are computed
                # from an input quantized to ~8 mantissa bits, which adds a few
                # percent of noise to `flux` in particular.
                S_log = src.float() * denorm_scale + self.cfg.prosody_denorm_mean

                prosody = compute_prosody(
                    S_log,
                    valid_time,
                    n_time,
                    patch_frames,
                    self.cfg.prosody_norm,
                    self.cfg.prosody_corpus_mean,
                    self.cfg.prosody_corpus_std,
                )                                                  # (B,3,64)
                prosody_bt = prosody.transpose(1, 2)               # (B,64,3)

                # BOTH source-derived tensors need the clone_batch repeat before
                # they meet masked_b, which is (B*clone_batch, 512).
                if self.cfg.clone_batch > 1:
                    prosody_bt = prosody_bt.repeat_interleave(self.cfg.clone_batch, 0)
                    valid_time = valid_time.repeat_interleave(self.cfg.clone_batch, 0)

                # The target at time t is a function of the whole mel column at t,
                # so a partially-visible column is readable rather than inferable.
                # Restrict to columns whose every freq-patch is masked.
                col_t = masked_b.view(-1, n_time, n_freq).all(-1) & valid_time   # (B*clone,64)

                # A row with no visible column cannot be interpolated, so it would
                # feed an arbitrary constant into the r2_interp baseline and bias
                # the very comparison this feature is judged on. Drop those rows
                # from the loss too, so loss and diagnostic see identical positions.
                anchor = (~masked_b.view(-1, n_time, n_freq).all(-1)) & valid_time
                col_t = col_t & anchor.any(-1, keepdim=True)

                col = col_t.repeat_interleave(n_freq, dim=-1)                    # (B*clone,512)

                prosody_target = prosody_bt.repeat_interleave(n_freq, dim=1)[col]
                sel = col[masked_b]

            # d2v_loss SUMS over the feature dim after applying `scale`, so the
            # per-dim count never cancels and prosody needs an explicit correction
            # to sit at parity with recon. The factor depends on which scale branch
            # d2v_loss takes (:1233-1251):
            #
            #   loss_scale is None -> scale = 1/sqrt(D)
            #       recon   = (1/16)*256*e = 16*e
            #       prosody = (1/sqrt3)*3*e = sqrt(3)*e     -> ratio sqrt(256/3) ~ 9.24
            #   loss_scale = c     -> scale = c for BOTH terms
            #       recon   = c*256*e
            #       prosody = c*3*e                         -> ratio 256/3 ~ 85.3
            #
            # The shared constant does NOT remove the imbalance -- it removes the
            # sqrt, making it strictly worse. Applied at the loss term rather than
            # via cfg.loss_scale, which is global and would rescale recon and d2v
            # too.
            recon_dim = patch_frames ** 2 * feature_extractor.modality_cfg.in_chans
            dim_parity = (recon_dim / 3) if self.loss_scale is not None else math.sqrt(recon_dim / 3)

            # An unlucky mask draw or a batch of very short utterances can leave no
            # fully-masked valid column at all. d2v_loss would reduce an empty
            # tensor to NaN and poison every gradient, but simply skipping the term
            # would leave prosody_proj without a gradient -- which DDP rejects
            # ("Expected to have finished reduction in the prior iteration") when
            # only some ranks hit the empty case. Emit an explicit zero that still
            # touches the parameter.
            if prosody_target.numel() > 0:
                pred = self.prosody_proj(xs[0][sel])
                prosody_loss = self.d2v_loss(pred, prosody_target.float()) * dim_parity
            else:
                pred = None
                prosody_loss = self.prosody_proj(xs[0][:1]).sum() * 0.0

            result["losses"]["prosody"] = (
                prosody_loss * self.cfg.prosody_loss * sample_size
            )

            # Triviality diagnostic. If the model cannot beat linear interpolation
            # from the visible columns, the task is solvable without learning
            # structure and will not reshape the encoder at any lambda. Only the
            # GAP is diagnostic -- a low r2_model alone is expected, since a smooth
            # 3-dim target has lower residual than 256-dim texture.
            #
            # Gated to log steps: the interpolation baseline and the subset assert
            # cost two scans plus a GPU->CPU sync, and nothing reads the numbers
            # in between.
            #
            # The gate must be RANK-UNIFORM. num_updates is synced, but
            # `pred is not None` is per-rank data-dependent (an unlucky mask draw or
            # a batch of very short utterances leaves no fully-masked valid column).
            # model_criterion.py:98 only writes a log key when it is present in
            # net_output, and _fast_stat_sync_sum (trainer.py:1442) derives its key
            # list from logging_outputs[0] LOCALLY on each rank before
            # all_reduce_dict. Ranks disagreeing on the key set therefore all-reduce
            # different-sized dicts -> NCCL hang. So emit all four keys whenever the
            # step is a diagnostic step, with zero sentinels when this rank had
            # nothing to score. prosody_frac_cols == 0 is what marks such a rank;
            # zeros (not NaN) because the criterion sums across ranks and one NaN
            # would poison every subsequent average.
            if self.num_updates % max(1, self.cfg.prosody_diag_interval) == 0:
                with torch.no_grad():
                    # Drift guard on the de-normalization constants. prosody_denorm_
                    # mean/std duplicate values that raw_audio_dataset.py:407-408
                    # sets INLINE, overwriting its own constructor arguments -- so
                    # nothing links the two, and a change to either goes unnoticed.
                    #
                    # The failure is silent and worse than a wrong scale: pad
                    # detection looks for frames sitting at exactly
                    # -mean/(2*std) ~ 0.46706, so mismatched constants match no
                    # frame, valid_time comes back all-true, and the pad plateau --
                    # which Step 0 recovers as a region LOUDER than quiet speech --
                    # is folded into every descriptor and into the corpus stats.
                    #
                    # At target_length=1024 (10.24s) an emotion corpus pads heavily,
                    # so a batch with NO detected padding anywhere is near-certain
                    # evidence of drift. Warn rather than raise: a corpus of
                    # uniformly long utterances would be a legitimate zero.
                    frac_padded = (valid_time.sum(-1) < n_time).float().mean()
                    result["prosody_frac_padded"] = frac_padded
                    if frac_padded == 0 and not getattr(self, "_prosody_pad_warned", False):
                        self._prosody_pad_warned = True
                        logger.warning(
                            "prosody: no padding detected in any utterance of this batch. "
                            "Expected heavy padding at target_length=%d. Check that "
                            "prosody_denorm_mean/std (%.4f/%.4f -> pad_value %.5f) still "
                            "match raw_audio_dataset.py:407-408; if they have drifted, "
                            "padded frames are silently contaminating the prosody target.",
                            n_time * patch_frames,
                            self.cfg.prosody_denorm_mean,
                            self.cfg.prosody_denorm_std,
                            -self.cfg.prosody_denorm_mean / (self.cfg.prosody_denorm_std * 2),
                        )

                    # ModelCriterion.reduce_metrics sums across ranks and divides
                    # by _world_size, so a rank contributing zero sentinels drags
                    # the mean down. r2_model and r2_interp are both scaled by
                    # (N-k)/N with k empty ranks, so the GAP -- the only thing the
                    # triviality decision reads -- is compressed toward zero, i.e.
                    # biased toward "the task is trivial, abandon it".
                    #
                    # This key averages to exactly (N-k)/N, so the true gap is
                    # (reported r2_model - reported r2_interp) / prosody_diag_frac.
                    # prosody_frac_cols cannot serve this purpose: it is averaged
                    # too, so k is not recoverable from it.
                    result["prosody_diag_frac"] = prosody_target.new_ones(()) if pred is not None \
                        else prosody_target.new_zeros(())

                    if pred is None:
                        # Sentinels only -- must NOT return early, or this rank
                        # would also skip the d2v/state keys emitted below and
                        # reintroduce the divergence from the other direction.
                        result["prosody_r2_model"] = prosody_target.new_zeros(())
                        result["prosody_r2_interp"] = prosody_target.new_zeros(())
                        result["prosody_target_var"] = prosody_target.new_zeros(3)
                        result["prosody_frac_cols"] = prosody_target.new_zeros(())
                    else:
                        # col is a subset of masked_b -- guaranteed by .all(-1)
                        # above, and load-bearing for these two lengths to match.
                        assert sel.sum() == col.sum()
                        interp = prosody_interp_baseline(prosody_bt, anchor)
                        interp = interp.repeat_interleave(n_freq, dim=1)[col]
                        result["prosody_r2_model"] = _r2(pred.float(), prosody_target.float())
                        result["prosody_r2_interp"] = _r2(interp.float(), prosody_target.float())
                        # per-descriptor, not pooled: the three are z-scored
                        # separately, so a pooled scalar cannot show one of them
                        # collapsing. The criterion expands this into _0/_1/_2
                        # (model_criterion.py:102-104) -- so the sentinel above must
                        # also be length 3 or the key COUNT would differ per rank.
                        # Inside the else on purpose: var(dim=0) over an empty
                        # (0, 3) target is NaN, and one NaN summed across ranks
                        # poisons every later average.
                        result["prosody_target_var"] = prosody_target.float().var(dim=0)
                        result["prosody_frac_cols"] = col_t.float().mean()

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                reg_loss = self.d2v_loss(x, y)
                assert reg_loss > 0, f"reg_loss must be positive, got {reg_loss}"
                n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss

        # compute state for logs
        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
            if self.ema is not None and hasattr(self.ema, "logs"):
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x

    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        assert x.size(-1) == y.size(-1), "d2v_loss: feature dim mismatch"
        y = y.view(-1, y.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale
        reg_loss = reg_loss.sum(dim=-1).mean()

        return reg_loss
    
    # average top-k layers output from teacher model
    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        self.ema = None
        self.cfg.clone_batch = 1
        self.recon_proj = None
        self.prosody_proj = None
        self.student_dino_head = None
        self.teacher_dino_head = None
        self.dino_loss_fn = None

        if not keep_decoder:
            self.shared_decoder = None

        modality = modality.lower() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if modality is not None and k.lower() != modality:
                del self.modality_encoders[k]
            else:
                self.modality_encoders[k].remove_pretraining_modules(
                    keep_decoder=keep_decoder
                )
                if not keep_decoder:
                    self.modality_encoders[k].decoder = None
