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

CAVEAT -- transductive scaling. The z-score written to disk is fit over all five
sessions, but eval_downstream_iemocap.py holds one session out per fold, so each
fold's test statistics are in its own training features. This is deliberate and
judged acceptable: the scaling uses NO labels and is a per-dimension affine, which
a linear probe with a learned first layer can absorb, so it cannot manufacture
class-discriminative signal -- it can only affect optimization conditioning. It is
still not zero. If the decision this probe drives lands near the floor, re-run
with --no_zscore (leak-free) before concluding either way, and report which
variant the decision was made on.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:                      # optional: the probes are useful without it
    import wandb
except Exception:
    wandb = None

from baselines.models.pretrain_eat import (
    PROSODY_CANDIDATES,
    compute_prosody,
    compute_prosody_candidates,
    prosody_valid_time,
)
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


def _loso_linear_acc(X, y, n_classes, alpha=1.0):
    """WA, UA and per-class recall of a closed-form linear classifier, 5-fold LOSO.

    Ridge regression onto one-hot targets, argmax at test time. Deterministic,
    dependency-free, and seconds to run -- this is an *indicator* for comparing
    feature subsets against each other, NOT a replacement for the real probe in
    eval_downstream_iemocap.py (which is an MLP with a different optimizer). Read
    the differences between rows, not the absolute values.

    Reports the same triple the real eval does, because WA alone would mislead on
    exactly the descriptor most at risk. Centroid is the only valence-axis
    descriptor, and IEMOCAP's ang/hap contrast is a valence pair at comparable
    arousal -- a feature that helps one class pair barely moves overall accuracy
    while clearly moving per-class recall. Judging it on WA would read "dead
    weight" when UA says otherwise.

    Note the floors differ: WA's is the majority-class rate, UA's is 1/n_classes
    (a majority classifier scores 100% on one class and 0% on the rest).
    """
    bounds = np.cumsum([0] + SESSION_SIZES)
    Y = np.eye(n_classes)[y]
    correct = total = 0
    cls_correct = np.zeros(n_classes, dtype=np.int64)
    cls_total = np.zeros(n_classes, dtype=np.int64)
    for i in range(5):
        te = np.zeros(X.shape[0], dtype=bool)
        te[bounds[i]:bounds[i + 1]] = True
        tr = ~te

        # Standardize per fold on TRAINING statistics only. Both halves matter:
        #
        # scale -- a single `alpha` across dims of wildly different magnitude
        #   penalizes them unequally. Under --no_zscore the raw summary dims span
        #   ~4 orders of magnitude (centroid ~1e2, log-energy ~1e1, slope ~1e-2),
        #   so an unscaled ridge would rank descriptors by numeric size rather
        #   than information -- and this table is what the "does flux earn its
        #   place" decision is read off.
        # leak  -- fitting on train rows only. This is only actually leak-free
        #   because callers pass the PRE-z-score matrix; standardizing per fold
        #   on top of a corpus-wide z-score would not undo it, and the ablation
        #   would silently inherit the same transductive caveat as the written
        #   features.
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Xs = (X - mu) / sd

        ym = Y[tr].mean(0, keepdims=True)
        Xc = Xs[tr]
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y[tr] - ym))
        pred = (Xs[te] @ W + ym).argmax(1)
        correct += int((pred == y[te]).sum())
        total += int(te.sum())
        for c in range(n_classes):
            m = y[te] == c
            cls_total[c] += int(m.sum())
            cls_correct[c] += int((pred[m] == c).sum())

    recall = np.where(cls_total > 0, cls_correct / np.maximum(cls_total, 1), np.nan)
    return {
        "wa": correct / total,
        "ua": float(np.nanmean(recall)),
        "recall": recall,
    }


def descriptor_ablation(X, names, labels, floor, candidates=()):
    """Per-descriptor and leave-one-out accuracy, computed in this same pass.

    Answers "does each descriptor earn its place?" without a second invocation or
    a flag to remember: the descriptors are CPU-only and the classifier is closed
    form, so the whole table costs well under a second.

    `candidates` are probe-only descriptors not in the training target. They get
    their own rows plus one-for-one swap rows, so the question "would alpha ratio
    be a better third descriptor than centroid?" is answered from data rather than
    from citation counts. X/names must cover training descriptors first, then
    candidates.
    """
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    every = list(DESCRIPTORS) + list(candidates)

    # Column blocks are computed arithmetically, NOT by name prefix. X/names are
    # descriptor-major with equal block size, and prefix matching silently breaks
    # the moment a candidate name extends an incumbent's: "flux_5k_mean"
    # .startswith("flux_") is True, so cols["flux"] would swallow flux_5k and
    # every subset row would be wrong while still printing plausible numbers.
    assert len(names) % len(every) == 0, (len(names), len(every))
    block = len(names) // len(every)
    cols = {d: list(range(i * block, (i + 1) * block)) for i, d in enumerate(every)}
    for i, d in enumerate(every):   # ordering assumption is load-bearing; check it
        assert names[i * block].startswith(d), (names[i * block], d)
    trained = [i for d in DESCRIPTORS for i in cols[d]]

    rows = [("all three", trained)]
    rows += [(f"{d} only", cols[d]) for d in DESCRIPTORS]
    rows += [(f"without {d}", [i for i in trained if i not in cols[d]]) for d in DESCRIPTORS]
    # candidates: standalone, then substituted for each incumbent in turn
    for c in candidates:
        rows.append((f"[{c}] only", cols[c]))
        for d in DESCRIPTORS:
            rows.append((f"[{c}] for {d}", [i for i in trained if i not in cols[d]] + cols[c]))

    ua_floor = 1.0 / len(classes)
    print("\n--- descriptor ablation (indicative linear probe, not the real one) ---")
    print(f"{'subset':<30} {'dims':>5} {'WA %':>7} {'UA %':>7} {'dWA':>7} {'dUA':>7}")
    base = None
    out = {}
    for name, idx in rows:
        if not idx:
            continue
        r = _loso_linear_acc(X[:, idx], y, len(classes))
        if base is None:
            base = r
        out[name] = {
            "wa": r["wa"], "ua": r["ua"],
            "recall": {c: (None if np.isnan(v) else float(v)) for c, v in zip(classes, r["recall"])},
        }
        dwa = "" if name == "all three" else f"{(r['wa'] - base['wa']) * 100:+7.2f}"
        dua = "" if name == "all three" else f"{(r['ua'] - base['ua']) * 100:+7.2f}"
        print(f"{name:<30} {len(idx):>5} {r['wa'] * 100:>7.2f} {r['ua'] * 100:>7.2f} {dwa:>7} {dua:>7}")

    print(f"floors: WA {floor * 100:.2f}% (majority class) | UA {ua_floor * 100:.2f}% (1/{len(classes)})")
    print("Read the DIFFERENCES between rows, not the absolute values. A descriptor")
    print("whose 'without' row is flat is not contributing on top of the others.")
    print("dUA is the sensitive one: a descriptor that helps a single class pair --")
    print(f"e.g. the valence contrast {classes[0]}/{classes[1] if len(classes) > 1 else '?'} at similar arousal -- moves UA while")
    print("leaving WA almost unchanged. Per-class recall is in the .ablation.json.")
    return out


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
        help="Skip the corpus z-score, removing the transductive-scaling caveat "
             "below. Use as the confirmatory run if the WA lands near the floor. "
             "Note BaseModel is Linear->ReLU with no input normalization, so raw "
             "descriptors (log-energy ~1e1, centroid ~1e2, flux ~1e1, very "
             "different spreads) condition the probe badly -- a low WA here is not "
             "by itself evidence the target is uninformative.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--sample_rate", type=int, default=16000,
                    help="Only used to derive mel band edges for the probe-only candidates.")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_mean", type=float, default=-4.268)
    ap.add_argument("--norm_std", type=float, default=4.569)
    ap.add_argument("--wandb_project", default=None,
                    help="Log the floor, the ablation table and the JSON artifact to W&B. "
                         "Without this the ablation exists only in stdout and a file on "
                         "pod-local disk.")
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
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
        # load_manifest_labels reads paths from the manifest and utt_ids/labels
        # from a separate file, checking only that the two have equal LENGTH -- it
        # never cross-checks their order. Every other guard below validates
        # utt_ids, i.e. the label file's ordering, while the spectrograms come from
        # paths. If the two files were sorted differently, each spectrogram would
        # be paired with another utterance's label, all the checks would pass, and
        # the probe would land at the majority-class floor -- reading as "the
        # direction is dead" under the decision rule this script exists to serve.
        mismatched = [
            (i, Path(p).stem, u) for i, (p, u) in enumerate(zip(paths, utt_ids)) if Path(p).stem != u
        ]
        if mismatched:
            i, stem, u = mismatched[0]
            raise ValueError(
                f"manifest and label file disagree on ordering at {len(mismatched)} of "
                f"{len(paths)} rows (first at index {i}: manifest has '{stem}', labels "
                f"have '{u}'). They are zipped positionally, so this would silently "
                "attach the wrong label to every spectrogram."
            )
    else:
        ap.error("provide --iemocap_root, or both --manifest and --labels")

    if not args.skip_session_check:
        # Counts alone are NOT sufficient. eval_downstream_iemocap.py slices fixed
        # positional ranges, so a manifest that is shuffled or interleaves sessions
        # can have exactly the right per-session totals and still put session-3
        # utterances inside fold 1's range. Check that each positional block
        # contains only its own session.
        sessions = [int(utt[4]) for utt in utt_ids]
        counts = [sessions.count(s) for s in range(1, 6)]
        assert counts == SESSION_SIZES, (
            f"per-session counts {counts} != {SESSION_SIZES}. eval_downstream_iemocap.py "
            "splits folds by position, so this would silently evaluate on wrong folds. "
            "Pass --skip_session_check only if you know the eval side matches."
        )
        bounds = np.cumsum([0] + SESSION_SIZES)
        for i in range(5):
            block = set(sessions[bounds[i]:bounds[i + 1]])
            assert block == {i + 1}, (
                f"positional block {i} (rows {bounds[i]}:{bounds[i + 1]}) contains "
                f"sessions {sorted(block)}, expected only {{{i + 1}}}. The utterance "
                "order is not session-contiguous, so the LOSO folds would mix "
                "sessions while every count still looks correct."
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
                # Probe-only candidates, appended for the ablation only. The
                # written feature file is sliced back to the training three
                # below, so the headline WA still answers "do the TRAINING
                # targets carry emotion" rather than a 6-descriptor superset.
                pc = compute_prosody_candidates(
                    S_log, valid_time, n_time, args.patch_size, args.sample_rate
                )
                x = torch.cat([x, summary_stats(pc, valid_time)], dim=-1)   # (B, 30)
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

    # Snapshot BEFORE the corpus z-score. The ablation standardizes per fold on
    # training rows anyway, so the corpus scaling is redundant for it -- and
    # passing the scaled matrix would mean the ablation inherits the same
    # transductive leak as the written features, which is exactly what the
    # per-fold fit was supposed to avoid. Feeding it the raw values makes the
    # ablation leak-free on the default path, not only under --no_zscore.
    X_raw = X.copy()

    if not args.no_zscore:
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
        # A constant dim (sd==0) carries no information; leave it at zero rather
        # than dividing by epsilon and amplifying float noise into a fake feature.
        keep = sd[0] > 1e-8
        X = np.where(keep, (X - mu) / np.where(keep, sd, 1.0), 0.0).astype(np.float32)
        if not keep.all():
            print(f"[warn] {int((~keep).sum())} constant dim(s) zeroed: {np.where(~keep)[0].tolist()}")

    # The summary variant hard-codes norm="none" (raw descriptors are the point --
    # absolute level is what it tests), so --prosody_norm is accepted but ignored
    # there. Record what was actually used, not what was passed: these JSONs are
    # the provenance trail, and a summary run invoked with --prosody_norm corpus
    # would otherwise be labelled `corpus` while containing raw statistics.
    norm_used = "none" if args.variant == "summary" else args.prosody_norm
    if args.variant == "summary" and args.prosody_norm != ap.get_default("prosody_norm"):
        print(f"[warn] --prosody_norm {args.prosody_norm} ignored for --variant summary "
              f"(raw descriptors by design); recording prosody_norm=none")

    # The candidates are for the ablation only -- the file eval_downstream_iemocap.py
    # probes must contain exactly the training target, or the headline WA would
    # answer a question about a 6-descriptor superset nobody trains on.
    n_trained = len(DESCRIPTORS) * len(STATS)
    # Three matrices, and mixing them up is easy:
    #   X_ablation  pre-z-score, full width -> descriptor_ablation ONLY, which
    #               standardizes per fold and would otherwise inherit the leak
    #   X_all       z-scored, full width    -> the _all file, Pre-flight B's target
    #   X           z-scored, training dims -> the file eval_downstream_iemocap.py
    #               probes; it feeds a Linear->ReLU MLP with no input
    #               normalization, so raw dims spanning 1e-2..1e2 would condition
    #               it badly and drive the headline WA toward the floor
    X_ablation = X_raw
    X_all = X
    X = X_all[:, :n_trained] if args.variant == "summary" else X_all

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{prefix}.npy", X.astype(np.float32))
    with open(f"{prefix}.lengths", "w") as f:
        f.writelines("1\n" for _ in range(X.shape[0]))
    with open(f"{prefix}.emo", "w") as f:
        f.writelines(e + "\n" for e in emo_entries)

    if args.variant == "summary":
        # X currently holds the training three THEN the probe-only candidates.
        names = [f"{d}_{s}" for d in DESCRIPTORS + PROSODY_CANDIDATES for s in STATS]
    else:
        names = [f"{d}_t{t}" for d in DESCRIPTORS for t in range(n_time)]
    with open(f"{prefix}.dims.json", "w") as f:
        json.dump({"variant": args.variant, "prosody_norm": norm_used,
                   "descriptors": DESCRIPTORS, "stats": STATS,
                   "dims": names[:X.shape[1]],
                   "ablation_only_dims": names[X.shape[1]:]}, f, indent=2)

    if args.variant == "summary" and X_all.shape[1] > X.shape[1]:
        # A second prefix carrying the candidates too, so Pre-flight B can report
        # encoder-decodability for them as well. Without it a candidate gets an
        # emotion-relevance score from the ablation but no R^2, and you would be
        # promoting it without knowing whether the encoder already encodes it --
        # which is exactly the pairing that decides whether a target teaches the
        # model anything new. NOT for eval_downstream_iemocap.py: probing it would
        # answer a question about a superset nobody trains on.
        alt = Path(f"{prefix}_all")
        np.save(f"{alt}.npy", X_all.astype(np.float32))
        with open(f"{alt}.lengths", "w") as f:
            f.writelines("1\n" for _ in range(X_all.shape[0]))
        with open(f"{alt}.emo", "w") as f:
            f.writelines(e + "\n" for e in emo_entries)
        with open(f"{alt}.dims.json", "w") as f:
            json.dump({"variant": args.variant, "prosody_norm": norm_used,
                       "descriptors": DESCRIPTORS + list(PROSODY_CANDIDATES),
                       "training_descriptors": DESCRIPTORS,
                       "stats": STATS, "dims": names,
                       "note": "training descriptors + probe-only candidates; for "
                               "preflight_prosody_r2.py, not for the downstream eval"},
                      f, indent=2)
        print(f"Wrote {alt}.npy  shape={X_all.shape}  (training + candidates, for Pre-flight B)")
    elif args.variant == "summary":
        # No candidates this run (e.g. one was promoted into DESCRIPTORS and the
        # list emptied). A stale _all from a previous descriptor set would still
        # satisfy the shell's `[ -f ]` check and send B against the wrong target
        # with the wrong dims.json -- and the utterance-order cross-check would
        # pass, because the corpus has not changed. Remove it.
        meta = Path(f"{prefix}_all.dims.json")
        ours = False
        if meta.exists():
            try:
                ours = "training_descriptors" in json.loads(meta.read_text())
            except Exception:
                ours = False
        if ours:
            # Only delete a prefix this script wrote -- `<name>_all` is a plausible
            # name for an unrelated extraction, and an [info] line is thin cover for
            # destroying one.
            for suf in (".npy", ".lengths", ".emo", ".dims.json"):
                stale = Path(f"{prefix}_all{suf}")
                if stale.exists():
                    stale.unlink()
                    print(f"[info] removed stale {stale.name} (no candidates this run)")
        elif meta.exists() or Path(f"{prefix}_all.npy").exists():
            print(f"[warn] {prefix}_all.* exists but was not written by this script "
                  "(no training_descriptors field); leaving it alone. Pre-flight B may "
                  "pick it up -- delete it yourself if it is stale.")

    vp = np.array(n_valid_patches)
    lab = np.array([e.split()[1] for e in emo_entries])
    uniq, cnt = np.unique(lab, return_counts=True)

    print(f"Wrote {prefix}.npy  shape={X.shape}  variant={args.variant}")
    print(f"valid patches per utterance: mean={vp.mean():.1f} min={vp.min()} max={vp.max()}")
    print(f"class counts: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"pooled majority-class rate  = {cnt.max() / cnt.sum() * 100:.2f}%")

    # The comparable floor is per-fold: eval_downstream_iemocap.py reports WA on a
    # held-out session, so the majority-class rate is computed per test fold and
    # averaged, not over the pooled corpus.
    #
    # This is only meaningful when the positional blocks really are the sessions,
    # which is exactly what --skip_session_check waives. Printing a confident floor
    # derived from the wrong partition would corrupt the go/no-go decision this
    # number exists to drive, so say so instead of guessing.
    if args.skip_session_check:
        print("MEAN PER-FOLD MAJORITY RATE = (not computed)")
        print("   ^ --skip_session_check waives the session-contiguity guarantee, so the")
        print("     fixed positional blocks are not known to be the LOSO folds. Compute the")
        print("     floor from whatever partition your eval side actually uses.")
    else:
        bounds = np.cumsum([0] + SESSION_SIZES)
        fold_major = [
            np.unique(lab[bounds[i]:bounds[i + 1]], return_counts=True)[1].max()
            / SESSION_SIZES[i]
            for i in range(5)
        ]
        print(
            f"MEAN PER-FOLD MAJORITY RATE = {np.mean(fold_major) * 100:.2f}%"
            f"  (per fold: {', '.join(f'{m*100:.1f}' for m in fold_major)})"
        )
        print("   ^ compare the probe's WA against this, not 25%")

        # Always run, no flag to remember. The question "does flux earn its place?"
        # is otherwise unanswerable with this tooling: the eval returns a single WA
        # over all 15 dims, and Pre-flight B measures decodability FROM the encoder,
        # which is the opposite direction.
        abl = descriptor_ablation(
            X_ablation, names, lab, float(np.mean(fold_major)),
            candidates=PROSODY_CANDIDATES if args.variant == "summary" else (),
        )
        with open(f"{prefix}.ablation.json", "w") as f:
            json.dump(
                {
                    "variant": args.variant,
                    "prosody_norm": norm_used,
                    "descriptors": DESCRIPTORS + (list(PROSODY_CANDIDATES)
                                                  if args.variant == "summary" else []),
                    "n_utterances": int(X_ablation.shape[0]),
                    "mean_per_fold_majority": float(np.mean(fold_major)),
                    "accuracy": abl,
                    "note": "closed-form ridge-to-one-hot LOSO classifier; indicative "
                            "only, for comparing subsets against each other. The "
                            "reported protocol is eval_downstream_iemocap.py.",
                },
                f,
                indent=2,
            )

    if args.wandb_project and wandb is not None and not args.skip_session_check:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name or f"preflight-descriptors-{args.variant}",
            job_type="preflight_a",
            config={"variant": args.variant, "prosody_norm": norm_used,
                    "descriptors": DESCRIPTORS, "candidates": list(PROSODY_CANDIDATES),
                    "n_utterances": int(X.shape[0]), "dims": int(X.shape[1])},
        )
        wandb.summary["floor/wa_majority"] = float(np.mean(fold_major))
        wandb.summary["floor/ua_chance"] = 1.0 / len(set(lab.tolist()))
        wandb.summary["valid_patches_mean"] = float(vp.mean())
        cols = ["subset", "dims", "wa", "ua", "d_wa", "d_ua"]
        base = abl.get("all three", {})
        tbl = wandb.Table(columns=cols)
        for name_, r in abl.items():
            tbl.add_data(name_, None, r["wa"], r["ua"],
                         r["wa"] - base.get("wa", r["wa"]), r["ua"] - base.get("ua", r["ua"]))
            key = name_.replace(" ", "_").replace("[", "").replace("]", "")
            wandb.summary[f"abl/{key}/wa"] = r["wa"]
            wandb.summary[f"abl/{key}/ua"] = r["ua"]
        wandb.log({"descriptor_ablation": tbl})
        art = wandb.Artifact(name=f"preflight-a-{args.variant}", type="preflight",
                             metadata={"descriptors": DESCRIPTORS})
        for suf in (".ablation.json", ".dims.json"):
            if Path(f"{prefix}{suf}").exists():
                art.add_file(f"{prefix}{suf}")
        run.log_artifact(art)
        wandb.finish()
    elif args.wandb_project and wandb is None:
        print("[warn] --wandb_project given but wandb is not importable; skipping W&B")

    print(f"\nNext: eval_downstream_iemocap.py --feat_prefix {prefix}")


if __name__ == "__main__":
    main()
