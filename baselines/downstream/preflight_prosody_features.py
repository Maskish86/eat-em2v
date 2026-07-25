#!/usr/bin/env python3
"""
Pre-flight A (see prosody-multitask-plan.md): do the three prosody descriptors
carry emotion at all, before any pretraining run is spent?

Writes the descriptors themselves as IEMOCAP "features" in the emotion2vec cached
format (<prefix>.npy/.lengths/.emo), one frame per utterance, so that
`eval_downstream_iemocap.py` runs over them **unchanged**: load_ssl_features uses
min_length=1 and BaseModel's masked mean over a single frame is the identity.

Two variants, because they answer different questions:

  summary  -- 15 dims: per-descriptor mean/std/min/max/slope over valid patches,
              computed from RAW descriptors. Absolute level is retained, so this
              tests whether level+shape carry emotion.
  contour  -- 192 dims: the actual training target (3 x 64) under --prosody_norm,
              flattened. This tests whether the thing the model is literally
              trained to predict carries emotion.

Decision rule (plan, "Pre-flight A"):
  both at the majority-class rate      -> stop, the direction is dead cheaply
  contour above floor, summary at floor -> emotion is in the contour; expect
                                           prosody_norm=instance to win
  both above floor                      -> target is emotion-correlated, level
                                           included; corpus mode worth the job

Report against the majority-class rate under the same folds -- NOT 25%. IEMOCAP's
4-class merge (exc+hap) is imbalanced.

The descriptors are imported from pretrain_eat.py rather than reimplemented, so
Step 0 (de-normalization), the flux t=0 convention and the floor(n_frames/16)
patch rule cannot drift between this probe and the training path.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.models.pretrain_eat import compute_prosody, prosody_valid_time
from baselines.downstream.extract_eat_iemocap_features import (
    IemocapSpecDataset,
    load_iemocap_root,
    load_manifest_labels,
)

DESCRIPTORS = ["log_energy", "centroid", "flux"]
STATS = ["mean", "std", "min", "max", "slope"]

# eval_downstream_iemocap.py splits folds by POSITION, not by utterance id
# (SESSION_SIZES there). This probe must therefore emit utterances in exactly the
# order extract_eat_iemocap_features.py does -- session-ordered, shuffle=False.
SESSION_SIZES = [1085, 1023, 1151, 1031, 1241]


def summary_stats(p, valid_time):
    """Per-descriptor mean/std/min/max/slope over valid patches only.

    p:          (B, 3, n_time) raw descriptors
    valid_time: (B, n_time) bool
    returns:    (B, 15) ordered descriptor-major: [energy_mean, energy_std, ...]
    """
    B, D, T = p.shape
    vt = valid_time.unsqueeze(1).to(p.dtype)                  # (B, 1, T)
    cnt = vt.sum(-1, keepdim=True).clamp(min=1.0)             # (B, 1, 1)

    mean = (p * vt).sum(-1, keepdim=True) / cnt
    var = (((p - mean) ** 2) * vt).sum(-1, keepdim=True) / cnt
    std = (var + 1e-8).sqrt()

    # min/max must ignore padded patches, which compute_prosody has zeroed --
    # a zero is a perfectly plausible descriptor value, so masking with +-inf is
    # required rather than relying on the zeros being extreme.
    neg_inf = torch.full_like(p, float("-inf"))
    pos_inf = torch.full_like(p, float("inf"))
    vmask = vt.bool().expand_as(p)
    mx = torch.where(vmask, p, neg_inf).amax(-1, keepdim=True)
    mn = torch.where(vmask, p, pos_inf).amin(-1, keepdim=True)

    # Least-squares slope over the valid prefix, in units of descriptor/patch.
    idx = torch.arange(T, device=p.device, dtype=p.dtype).view(1, 1, T)
    t_mean = (idx * vt).sum(-1, keepdim=True) / cnt
    dt = (idx - t_mean) * vt
    slope = (dt * (p - mean)).sum(-1, keepdim=True) / ((dt ** 2).sum(-1, keepdim=True) + 1e-8)

    out = torch.cat([mean, std, mn, mx, slope], dim=-1)       # (B, 3, 5)
    return out.reshape(B, D * len(STATS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="Path to IEMOCAP manifest (tsv with root on first line).")
    ap.add_argument("--labels", help="Path to labels file (lines: <utt_id> <label>).")
    ap.add_argument("--iemocap_root", help="IEMOCAP root with Session1..Session5.")
    ap.add_argument("--output_prefix", required=True, help="Prefix for output (.npy/.lengths/.emo).")
    ap.add_argument("--variant", choices=["summary", "contour"], default="summary")
    ap.add_argument(
        "--prosody_norm",
        default="instance",
        choices=["instance", "corpus", "none"],
        help="contour variant only; must match the training config being probed for.",
    )
    ap.add_argument("--corpus_mean", type=float, nargs=3, default=None)
    ap.add_argument("--corpus_std", type=float, nargs=3, default=None)
    ap.add_argument(
        "--no_zscore",
        action="store_true",
        help="Skip the corpus z-score. Only for inspecting raw values -- the linear "
             "probe conditions badly on unscaled descriptors (log-energy ~1e1, "
             "centroid ~1e2, flux ~1e1 with very different spreads).",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_mean", type=float, default=-4.268)
    ap.add_argument("--norm_std", type=float, default=4.569)
    ap.add_argument(
        "--skip_session_check",
        action="store_true",
        help="Bypass the SESSION_SIZES assertion. Only for non-standard subsets -- "
             "eval_downstream_iemocap.py splits folds by position, so a mismatch "
             "means the folds are silently wrong.",
    )
    args = ap.parse_args()

    if args.iemocap_root:
        paths, utt_ids, labels = load_iemocap_root(args.iemocap_root)
    elif args.manifest and args.labels:
        paths, utt_ids, labels = load_manifest_labels(args.manifest, args.labels)
    else:
        ap.error("provide --iemocap_root, or both --manifest and --labels")

    if not args.skip_session_check:
        counts = [0] * 5
        for utt in utt_ids:
            counts[int(utt[4]) - 1] += 1
        assert counts == SESSION_SIZES, (
            f"per-session counts {counts} != {SESSION_SIZES}. eval_downstream_iemocap.py "
            "splits folds by position, so this would silently evaluate on wrong folds. "
            "Pass --skip_session_check only if you know the eval side matches."
        )

    dataset = IemocapSpecDataset(
        paths,
        utt_ids,
        labels,
        target_length=args.target_length,
        n_mels=args.n_mels,
        patch_size=args.patch_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # load-bearing: folds are positional
        num_workers=args.num_workers,
        collate_fn=IemocapSpecDataset.collate_fn,
    )

    n_time = args.target_length // args.patch_size
    denorm_scale = args.norm_std * 2
    pad_value = -args.norm_mean / denorm_scale

    feats, emo_entries = [], []
    n_valid_patches = []

    for mels, _valid_blocks, utts, lbls in loader:
        src = mels.squeeze(1)                                   # (B, T, M), normalized
        with torch.no_grad():
            # Route B pad detection, on the normalized input, exactly as training
            # does -- floor(n_frames/16), not the ceil() this dataset's own
            # valid_time_blocks uses for the encoder readout.
            valid_time = prosody_valid_time(src, pad_value, n_time, args.patch_size)
            S_log = src.float() * denorm_scale + args.norm_mean   # Step 0

            if args.variant == "summary":
                p = compute_prosody(S_log, valid_time, n_time, args.patch_size, "none")
                x = summary_stats(p, valid_time)                  # (B, 15)
            else:
                p = compute_prosody(
                    S_log,
                    valid_time,
                    n_time,
                    args.patch_size,
                    args.prosody_norm,
                    args.corpus_mean,
                    args.corpus_std,
                )
                x = p.reshape(p.size(0), -1)                      # (B, 192)

        feats.append(x.cpu().float().numpy())
        n_valid_patches.extend(valid_time.sum(-1).cpu().tolist())
        for utt, lbl in zip(utts, lbls):
            emo_entries.append(f"{utt} {lbl}")

    X = np.concatenate(feats, axis=0)                             # (N, D)
    assert X.shape[0] == len(utt_ids), f"{X.shape[0]} != {len(utt_ids)}"

    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(
            f"{len(bad)} non-finite descriptor values (first at row {bad[0][0]}, "
            f"col {bad[0][1]}). An utterance with zero valid patches would do this."
        )

    if not args.no_zscore:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
        # A constant dim (sd==0) carries no information; leave it at zero rather
        # than dividing by epsilon and amplifying float noise into a fake feature.
        keep = sd[0] > 1e-8
        X = np.where(keep, (X - mu) / np.where(keep, sd, 1.0), 0.0).astype(np.float32)
        if not keep.all():
            print(f"[warn] {int((~keep).sum())} constant dim(s) zeroed: {np.where(~keep)[0].tolist()}")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{prefix}.npy", X.astype(np.float32))
    with open(f"{prefix}.lengths", "w") as f:
        f.writelines("1\n" for _ in range(X.shape[0]))
    with open(f"{prefix}.emo", "w") as f:
        f.writelines(e + "\n" for e in emo_entries)

    if args.variant == "summary":
        names = [f"{d}_{s}" for d in DESCRIPTORS for s in STATS]
    else:
        names = [f"{d}_t{t}" for d in DESCRIPTORS for t in range(n_time)]
    with open(f"{prefix}.dims.json", "w") as f:
        json.dump({"variant": args.variant, "prosody_norm": args.prosody_norm, "dims": names}, f, indent=2)

    vp = np.array(n_valid_patches)
    lab = np.array([e.split()[1] for e in emo_entries])
    uniq, cnt = np.unique(lab, return_counts=True)

    # The comparable floor is per-fold: eval_downstream_iemocap.py reports WA on a
    # held-out session, so the majority-class rate is computed per test fold and
    # averaged, not over the pooled corpus.
    bounds = np.cumsum([0] + SESSION_SIZES)
    fold_major = []
    for i in range(5):
        fold = lab[bounds[i]:bounds[i + 1]]
        if len(fold):
            fold_major.append(np.unique(fold, return_counts=True)[1].max() / len(fold))

    print(f"Wrote {prefix}.npy  shape={X.shape}  variant={args.variant}")
    print(f"valid patches per utterance: mean={vp.mean():.1f} min={vp.min()} max={vp.max()}")
    print(f"class counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"pooled majority-class rate  = {cnt.max() / cnt.sum() * 100:.2f}%")
    print(
        f"MEAN PER-FOLD MAJORITY RATE = {np.mean(fold_major) * 100:.2f}%"
        f"  (per fold: {', '.join(f'{m*100:.1f}' for m in fold_major)})"
    )
    print("   ^ compare the probe's WA against this, not 25%")
    print(f"\nNext: eval_downstream_iemocap.py --feat_prefix {prefix}")


if __name__ == "__main__":
    main()
