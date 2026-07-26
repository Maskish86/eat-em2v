#!/usr/bin/env python3
"""
Offline corpus statistics for `prosody_norm: corpus` (see prosody-multitask-plan.md,
"Level normalization").

Emits the six numbers -- 3 per-descriptor means, 3 stds -- that `corpus` mode
subtracts. Without them, corpus mode asserts at construction.

WHY THIS IS NOT A THROWAWAY SCRIPT. This is the third pad-leak site in the plan.
Padding is applied *before* the dataset's global normalization, so padded frames
recover to a constant plateau LOUDER than genuinely quiet speech. If these
statistics are accumulated over padded frames, the stored constants are wrong for
every subsequent run, the target is uniformly offset and mis-scaled forever, and
nothing in the training curves reveals it.

Two defences, both load-bearing:
  1. This script imports `compute_prosody` / `prosody_valid_time` from
     pretrain_eat.py rather than reimplementing them, so Step 0, the flux t=0
     convention and the floor(n_frames/16) patch rule cannot drift from training.
  2. Statistics accumulate over VALID patches only, via the same valid_time mask
     the loss uses.

The output records the manifest and the git commit alongside the numbers: they are
corpus-specific and silently invalid if the pretraining data changes.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from baselines.data.raw_audio_dataset import FileAudioDataset
from baselines.models.pretrain_eat import compute_prosody, prosody_valid_time

DESCRIPTORS = ["log_energy", "centroid", "flux"]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Pretraining manifest (.tsv, root on first line).")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--target_length", type=int, default=1024)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_mean", type=float, default=-4.268)
    ap.add_argument("--norm_std", type=float, default=4.569)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_utts", type=int, default=None, help="Stop after N utterances (smoke test only).")
    # These MUST match what mae_image_dataset.py:126-142 passes for the run whose
    # constants these become. They are surfaced as flags rather than hardcoded
    # because the task exposes them (h5_format, downsr_16hz) -- but the defaults
    # are the training values, not this script's convenience.
    ap.add_argument("--sample_rate", type=int, default=32000, help="mae_image_dataset passes 32000.")
    ap.add_argument("--max_sample_size", type=int, default=325000)
    ap.add_argument("--min_sample_size", type=int, default=10000,
                    help="Training skips shorter utterances; including them here biases the constants.")
    ap.add_argument("--h5_format", action="store_true", help="Set if task.h5_format is set for the run.")
    ap.add_argument("--no_downsr_16hz", action="store_true", help="Unset if task.downsr_16hz is false for the run.")
    args = ap.parse_args()

    # The dataset must be built the way TRAINING builds it
    # (baselines/data/mae_image_dataset.py:126-142), not the way that is
    # convenient here -- these constants are baked into every subsequent
    # prosody_norm=corpus run, and a divergence is invisible in the training
    # curves. Differences that matter:
    #   min_sample_size  training skips <10000-sample utterances; if this script
    #                    includes them the mean/std describe a different corpus
    #   h5_format        on an h5 manifest, False makes every read fall through to
    #                    sf.read("<root>/10.h5/x.wav"), retry 3x and raise
    #   sample_rate      postprocess() raises on a mismatch unless downsr_16hz
    #                    rewrites it first
    #
    # train_mode stays 'train'. It has exactly three uses (raw_audio_dataset.py:355,
    # :385, :412): it gates the HDF5 reader, `roll_mag_aug`, and `noise`. Setting
    # 'valid' to suppress the augmentations would ALSO disable the HDF5 reader, so
    # an h5 manifest would fall through to sf.read("<root>/10.h5/x.wav"), retry 3x
    # and raise before producing a single statistic. The augmentations are instead
    # switched off by their own flags below, which is sufficient on its own --
    # additive noise would destroy the exact-constancy the pad detection relies on,
    # and the time-axis roll would stop padding being trailing at all.
    dataset = FileAudioDataset(
        manifest_path=args.manifest,
        sample_rate=args.sample_rate,
        max_sample_size=args.max_sample_size,
        min_sample_size=args.min_sample_size,
        shuffle=False,
        pad=False,
        normalize=True,
        num_buckets=0,
        compute_mask=False,
        h5_format=args.h5_format,
        downsr_16hz=not args.no_downsr_16hz,
        wav2fbank=True,
        target_length=args.target_length,
        roll_mag_aug=False,
        noise=False,
        train_mode="train",
    )
    # These two are hardcoded constants, not knobs that could be made to match a
    # training run -- and deliberately so:
    #   noise         breaks Route B pad detection outright (destroys the exact
    #                 constancy of the pad plateau)
    #   roll_mag_aug  applies a random gain, i.e. randomizes the absolute log-energy
    #                 these constants exist to center. prosody_norm=corpus asserts
    #                 task.roll_aug=False for the same reason (pretrain_eat.py), so
    #                 a run these statistics are valid for cannot have it on.
    # If a run needs roll_aug, it must use prosody_norm=instance, which subtracts
    # the per-utterance mean and needs no constants at all.
    assert not dataset.noise, "noise must be off for corpus statistics"
    assert not dataset.roll_mag_aug, "roll_mag_aug must be off for corpus statistics"
    print(f"{len(dataset)} utterances after min_sample_size={args.min_sample_size} filtering "
          f"({len(getattr(dataset, 'skipped_indices', []))} skipped)")

    n_time = args.target_length // args.patch_size
    denorm_scale = args.norm_std * 2
    pad_value = -args.norm_mean / denorm_scale

    # float64 accumulators: this runs over the whole pretraining corpus, and a
    # float32 running sum of ~10^8 terms loses low-order bits that matter for the
    # variance.
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = np.zeros(3, dtype=np.float64)
    n_utts = 0
    n_empty = 0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: torch.stack([x["source"].squeeze(0) for x in b], dim=0),
    )

    with torch.no_grad():
        for src in loader:                                       # (B, T, M) normalized
            valid_time = prosody_valid_time(src, pad_value, n_time, args.patch_size)
            S_log = src.float() * denorm_scale + args.norm_mean  # Step 0

            # norm="none": the raw descriptors are exactly what the stats describe.
            # Normalizing here would make the output the statistics of an already
            # normalized quantity -- i.e. 0 and 1, silently.
            p = compute_prosody(S_log, valid_time, n_time, args.patch_size, "none")

            vt = valid_time.unsqueeze(1).to(p.dtype)             # (B, 1, n_time)
            pv = (p * vt).double()
            total += pv.sum(dim=(0, 2)).cpu().numpy()
            total_sq += (pv * pv).sum(dim=(0, 2)).cpu().numpy()
            count += vt.expand_as(p).sum(dim=(0, 2)).double().cpu().numpy()

            n_empty += int((valid_time.sum(-1) == 0).sum())
            n_utts += src.shape[0]
            if n_utts % (args.batch_size * 50) == 0:
                print(f"  {n_utts} utterances...", flush=True)
            if args.max_utts and n_utts >= args.max_utts:
                break

    if count.min() == 0:
        raise ValueError("no valid patches found -- pad detection or the manifest is wrong")

    mean = total / count
    var = total_sq / count - mean ** 2
    if (var < 0).any():
        raise ValueError(f"negative variance from catastrophic cancellation: {var}")
    std = np.sqrt(var)

    out = {
        "manifest": args.manifest,
        "git_commit": git_commit(),
        "n_utterances": n_utts,
        "n_utterances_with_no_valid_patch": n_empty,
        "n_valid_patches": int(count[0]),
        "target_length": args.target_length,
        "patch_size": args.patch_size,
        "descriptors": DESCRIPTORS,
        "prosody_corpus_mean": mean.tolist(),
        "prosody_corpus_std": std.tolist(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{n_utts} utterances, {int(count[0])} valid patches"
          + (f", {n_empty} utterances had NO valid patch" if n_empty else ""))
    for i, d in enumerate(DESCRIPTORS):
        print(f"  {d:<12} mean={mean[i]:>10.4f}  std={std[i]:>10.4f}")
    print(f"\nWrote {args.output_json}\n")
    print("Paste into the model block of the pretraining config, and keep the manifest")
    print(f"name and commit ({out['git_commit'][:8]}) in a comment beside them:\n")
    print("  prosody_norm: corpus")
    print(f"  prosody_corpus_mean: [{', '.join(f'{v:.6f}' for v in mean)}]")
    print(f"  prosody_corpus_std: [{', '.join(f'{v:.6f}' for v in std)}]")


if __name__ == "__main__":
    main()
