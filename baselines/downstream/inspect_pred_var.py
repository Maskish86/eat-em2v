#!/usr/bin/env python3
"""
Diagnostic: locate where per-dimension variance collapses in the EAT-em2v
student->decoder pipeline, to explain pred_var << target_var.

No training. Loads a pretraining checkpoint WITH its EMA teacher + decoder
(unlike extract_eat_iemocap_features.load_eat, which strips them), runs the
same masked forward used in training on one batch, and reports
`compute_var` (per-dim std across examples, mean over dims) at every stage:

    student CNN local features
    student transformer block 0..11   (visible tokens)
    decoder output = pred_var          (masked positions)  <- the 0.44 metric
    teacher transformer block 0..11
    teacher block-average = target_var (masked positions)  <- the 0.98 metric

Where the number falls from ~1.0 toward ~0.44 is the layer to target.

Usage:
    python -m baselines.downstream.inspect_pred_var \
        --checkpoint /path/to/026784368_or_0ec956b2.pt \
        --em2v_cfg baselines/configs/pretrain_eat_em2v_style.yaml \
        --iemocap_root /path/to/IEMOCAP \
        --batch_size 16
"""

import argparse

import torch
from omegaconf import OmegaConf, open_dict

import fairseq

# Reuse the data pipeline and the fairseq registration/config plumbing that
# already works in the feature-extraction path.
from baselines.downstream.extract_eat_iemocap_features import (
    IemocapSpecDataset,
    load_iemocap_root,
    load_manifest_labels,
    _resolve_em2v_cfg,
    _get_cfg_node,
    _select_state_dict,
    _clear_fairseq_task,
    _clear_fairseq_model,
    _import_user_dir,
    _ensure_eat_registrations,
)
from baselines.utils.fairseq_compat import apply_fairseq_compat_patches
from torch.utils.data import DataLoader


def load_eat_em2v_with_teacher(checkpoint, device, em2v_cfg=None):
    """Like extract's load_eat(eat_em2v) but keeps EMA teacher + decoder so the
    training-time masked forward (and thus pred_var) can be reproduced."""
    apply_fairseq_compat_patches()
    _clear_fairseq_task("mae_image_pretraining", "MaeImagePretrainingTask")
    _clear_fairseq_model("data2vec_multi", "Data2VecMultiModel")
    _import_user_dir("eat_em2v")
    _ensure_eat_registrations("eat_em2v")

    state = torch.load(checkpoint, map_location="cpu")
    eat_cfg = _resolve_em2v_cfg(state, em2v_cfg)
    if eat_cfg is None:
        raise KeyError(
            "EAT-em2v checkpoint missing config; pass --em2v_cfg with the "
            "pretraining YAML."
        )
    if not OmegaConf.is_config(eat_cfg):
        eat_cfg = OmegaConf.create(
            vars(eat_cfg) if hasattr(eat_cfg, "__dict__") else eat_cfg
        )

    raw_task_cfg = _get_cfg_node(eat_cfg, "task")
    raw_model_cfg = _get_cfg_node(eat_cfg, "model")
    if raw_task_cfg is None or raw_model_cfg is None:
        raise KeyError("EAT-em2v config must include both 'task' and 'model'.")

    task_data = OmegaConf.to_container(raw_task_cfg, resolve=True)
    task_cfg = OmegaConf.create(
        {k: v for k, v in task_data.items() if not k.startswith("wandb_")}
    )

    # Keep skip_ema False and DO NOT drop ema_decay: we want the teacher built.
    model_data = OmegaConf.to_container(raw_model_cfg, resolve=True)
    # layer_decay/no_decay_blocks only affect the optimizer; harmless to keep,
    # but strip to avoid any build-time assumptions about optim overrides.
    for k in ("no_decay_blocks",):
        model_data.pop(k, None)
    model_data["skip_ema"] = False
    model_cfg = OmegaConf.create(model_data)

    task = fairseq.tasks.setup_task(task_cfg)
    model = fairseq.models.build_model(model_cfg, task)
    # student_state_dict is preferred; teacher (_ema) may be absent -> the
    # freshly built teacher is a copy of the loaded student, fine for a probe.
    missing = model.load_state_dict(_select_state_dict(state), strict=False)
    print(f"[load] missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}")

    model.eval()
    model.to(device)
    return model


def cv(model, z):
    """model.compute_var: per-dim std across examples, averaged over dims."""
    return model.compute_var(z.float().detach()).item()


def main():
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--em2v_cfg", default=None,
                    help="Optional; em2v checkpoints usually embed their config.")
    ap.add_argument("--iemocap_root")
    ap.add_argument("--manifest")
    ap.add_argument("--labels")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--target_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_mean", type=float, default=-4.268)
    ap.add_argument("--norm_std", type=float, default=4.569)
    ap.add_argument("--clone_batch", type=int, default=None,
                    help="Override cfg.clone_batch (lower = less memory).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_eat_em2v_with_teacher(args.checkpoint, device, args.em2v_cfg)
    if args.clone_batch is not None:
        model.cfg.clone_batch = args.clone_batch

    # --- register variance taps -------------------------------------------
    taps = {}

    def hook(name):
        def _h(_m, _i, out):
            z = out[0] if isinstance(out, tuple) else out
            taps[name] = cv(model, z)
        return _h

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(hook(f"student_blk_{i:02d}")))
    # teacher blocks: ema.model is the blocks ModuleList (ema_encoder_only=True)
    tm = model.ema.model
    teacher_blocks = tm if isinstance(tm, torch.nn.ModuleList) else tm.blocks
    for i, blk in enumerate(teacher_blocks):
        handles.append(blk.register_forward_hook(hook(f"teacher_blk_{i:02d}")))

    # --- data --------------------------------------------------------------
    if args.iemocap_root:
        paths, utt_ids, labels = load_iemocap_root(args.iemocap_root)
    else:
        paths, utt_ids, labels = load_manifest_labels(args.manifest, args.labels)
    dataset = IemocapSpecDataset(
        paths, utt_ids, labels,
        target_length=args.target_length, n_mels=args.n_mels,
        patch_size=args.patch_size, norm_mean=args.norm_mean, norm_std=args.norm_std,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, collate_fn=dataset.collate_fn)

    mels, _valid, _utts, _lbls = next(iter(loader))
    mels = mels.to(device)

    # --- masked forward (training path): fills result with pred_var/target_var
    result = model(mels, mask=True, features_only=False)

    for h in handles:
        h.remove()

    # --- report ------------------------------------------------------------
    print("\n=== per-dim std across examples (compute_var) ===")
    print(f"{'stage':<20} std")
    for k in sorted(taps):
        if k.startswith("student"):
            print(f"{k:<20} {taps[k]:.4f}")
    print("-" * 28)
    for k in sorted(taps):
        if k.startswith("teacher"):
            print(f"{k:<20} {taps[k]:.4f}")
    print("-" * 28)
    # decoder output (masked positions) and teacher target, straight from forward
    for k, v in result.items():
        if isinstance(k, str) and (k.startswith("pred_var") or k.startswith("target_var")):
            val = v.item() if torch.is_tensor(v) else v
            print(f"{k:<20} {val:.4f}   <- from forward()")

    print("\nRead: student_blk_* is visible-token std; pred_var is decoder "
          "output at masked positions. If student blocks stay ~1.0 and the "
          "drop appears only at pred_var, the decoder is the collapse point. "
          "If student blocks decline with depth, it is the encoder.")


if __name__ == "__main__":
    main()