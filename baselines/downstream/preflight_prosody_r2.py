#!/usr/bin/env python3
"""
Pre-flight B (see prosody-multitask-plan.md): how decodable is prosody from the
encoder's mean-pooled features?

Fits ridge regression from mean-pooled IEMOCAP features (whatever
extract_eat_iemocap_features.py wrote) to the 15-dim prosody summary written by
preflight_prosody_features.py --variant summary, and reports per-descriptor R^2
on held-out session folds.

Run it twice:

  1. NOW, on frozen EAT -- this is the baseline. Store the three numbers next to
     the 67.07% frozen-EAT reference in experiment-summary-dino-lr.md.
  2. AFTER a prosody pretraining run -- success requirement 2. The claim under
     test is that R^2 ROSE against (1), tracking the WA gain.

Expected direction is NOT "R^2 ~ 0" at baseline: EAT reconstructs spectrograms,
so some prosody is already decodable. The claim is that it rose, not that it
appeared from nothing.

Given the effect size this plan honestly expects (see the warning box in the
plan), this measurement -- not the WA delta -- is the primary deliverable: it has
the better signal-to-noise ratio and it reads directly on whether the prosody term
did what it was designed to do.

Ridge is solved in closed form with numpy; no sklearn dependency is added.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DESCRIPTORS = ["log_energy", "centroid", "flux"]
STATS = ["mean", "std", "min", "max", "slope"]
SESSION_SIZES = [1085, 1023, 1151, 1031, 1241]


def load_prefix(prefix):
    """Load emotion2vec-format cached features -> (list of (T, D) arrays, utt ids)."""
    prefix = Path(prefix)
    feats = np.load(f"{prefix}.npy")
    with open(f"{prefix}.lengths") as f:
        lengths = [int(line) for line in f if line.strip()]
    with open(f"{prefix}.emo") as f:
        utts = [line.split()[0] for line in f if line.strip()]

    assert sum(lengths) == feats.shape[0], (
        f"{prefix}: lengths sum to {sum(lengths)} but .npy has {feats.shape[0]} rows"
    )
    assert len(lengths) == len(utts), f"{prefix}: {len(lengths)} lengths vs {len(utts)} labels"

    offsets = np.cumsum([0] + lengths)
    return feats, offsets, utts


def mean_pool(feats, offsets):
    n = len(offsets) - 1
    out = np.zeros((n, feats.shape[1]), dtype=np.float64)
    for i in range(n):
        out[i] = feats[offsets[i]:offsets[i + 1]].mean(axis=0)
    return out


def ridge_fit_predict(X_tr, Y_tr, X_te, alpha):
    """Closed-form ridge with an intercept, fit on train, applied to test.

    Centering both sides lets the intercept stay out of the penalty, which
    matters here: the targets are z-scored per corpus, not per fold, so their
    per-fold means are not zero.
    """
    xm, ym = X_tr.mean(0, keepdims=True), Y_tr.mean(0, keepdims=True)
    Xc, Yc = X_tr - xm, Y_tr - ym
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    return (X_te - xm) @ W + ym


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat_prefix", required=True, help="Encoder features (from extract_eat_iemocap_features.py).")
    ap.add_argument("--prosody_prefix", required=True, help="Summary target (preflight_prosody_features.py --variant summary).")
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0],
                    help="Ridge penalties; the best is chosen per fold on an inner split of the training sessions.")
    ap.add_argument("--output_json", default=None, help="Where to write the numbers (default: <feat_prefix>.prosody_r2.json).")
    ap.add_argument("--label", default=None, help="Name for this encoder in the printout, e.g. 'frozen-EAT'.")
    args = ap.parse_args()

    feats, offsets, utts = load_prefix(args.feat_prefix)
    p_feats, p_offsets, p_utts = load_prefix(args.prosody_prefix)

    if utts != p_utts:
        first = next((i for i, (a, b) in enumerate(zip(utts, p_utts)) if a != b), None)
        raise ValueError(
            "utterance order differs between the two prefixes"
            + (f" (first mismatch at {first}: {utts[first]} vs {p_utts[first]})" if first is not None
               else f" (lengths {len(utts)} vs {len(p_utts)})")
            + ". Both must come from the same session-ordered, shuffle=False pass."
        )

    X = mean_pool(feats, offsets)
    Y = mean_pool(p_feats, p_offsets)  # 1 frame per utterance, so this is identity

    assert Y.shape[1] == len(DESCRIPTORS) * len(STATS), (
        f"expected a {len(DESCRIPTORS) * len(STATS)}-dim summary target, got {Y.shape[1]}. "
        "Pass --variant summary to preflight_prosody_features.py."
    )
    if sum(SESSION_SIZES) != X.shape[0]:
        print(f"[warn] {X.shape[0]} utterances != {sum(SESSION_SIZES)} expected; folds may not be sessions.")

    bounds = np.cumsum([0] + SESSION_SIZES)
    n_folds = sum(1 for i in range(5) if bounds[i + 1] <= X.shape[0])

    per_fold = []
    for i in range(n_folds):
        te = np.zeros(X.shape[0], dtype=bool)
        te[bounds[i]:bounds[i + 1]] = True
        tr = ~te

        # Pick alpha on a held-out slice of the TRAINING sessions -- never on the
        # test fold, which is the whole point of the LOSO protocol.
        #
        # Interleaved, not a positional tail. The data is session-contiguous, so
        # tr_idx[cut:] would be the last ~20% of the training rows -- i.e. the tail
        # of Session 5 for folds 1-4, and of Session 4 for fold 5. Alpha would then
        # be selected on one speaker-session (the same one four times out of five),
        # mid-session, rather than on a representative sample. Taking every 5th row
        # spreads the inner validation set across all four training sessions.
        tr_idx = np.where(tr)[0]
        inner_va = tr_idx[::5]
        inner_tr = np.setdiff1d(tr_idx, inner_va, assume_unique=True)
        best_alpha, best_score = args.alpha[0], -np.inf
        for a in args.alpha:
            pred = ridge_fit_predict(X[inner_tr], Y[inner_tr], X[inner_va], a)
            ss_res = ((pred - Y[inner_va]) ** 2).sum(0)
            ss_tot = ((Y[inner_va] - Y[inner_va].mean(0, keepdims=True)) ** 2).sum(0)
            score = (1 - ss_res / (ss_tot + 1e-12)).mean()
            if score > best_score:
                best_alpha, best_score = a, score

        pred = ridge_fit_predict(X[tr], Y[tr], X[te], best_alpha)
        ss_res = ((pred - Y[te]) ** 2).sum(0)
        ss_tot = ((Y[te] - Y[te].mean(0, keepdims=True)) ** 2).sum(0)
        per_fold.append({"fold": i + 1, "alpha": best_alpha, "r2": (1 - ss_res / (ss_tot + 1e-12))})

    r2 = np.stack([f["r2"] for f in per_fold])            # (folds, 15)
    r2_mean, r2_std = r2.mean(0), r2.std(0)

    names = [f"{d}_{s}" for d in DESCRIPTORS for s in STATS]
    grouped = {
        d: float(r2_mean[j * len(STATS):(j + 1) * len(STATS)].mean())
        for j, d in enumerate(DESCRIPTORS)
    }

    tag = args.label or Path(args.feat_prefix).name
    print(f"\nPre-flight B -- prosody decodability from mean-pooled features [{tag}]")
    print(f"{n_folds}-fold LOSO, ridge alpha per fold: {[f['alpha'] for f in per_fold]}\n")
    print(f"{'dim':<20} {'R2':>8} {'+-':>7}")
    for n, m, s in zip(names, r2_mean, r2_std):
        print(f"{n:<20} {m:>8.4f} {s:>7.4f}")
    print(f"\n{'per-descriptor R2':<20}")
    for d, v in grouped.items():
        print(f"  {d:<18} {v:>8.4f}")
    print(f"  {'OVERALL':<18} {r2_mean.mean():>8.4f}")
    print(
        "\nBaseline run: store these next to the 67.07% frozen-EAT reference.\n"
        "Post-training run: the claim is that these ROSE, tracking the WA gain."
    )

    out = Path(args.output_json) if args.output_json else Path(f"{args.feat_prefix}.prosody_r2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "label": tag,
                "feat_prefix": args.feat_prefix,
                "prosody_prefix": args.prosody_prefix,
                "n_folds": n_folds,
                "per_dim_r2_mean": dict(zip(names, r2_mean.tolist())),
                "per_dim_r2_std": dict(zip(names, r2_std.tolist())),
                "per_descriptor_r2": grouped,
                "overall_r2": float(r2_mean.mean()),
                "per_fold": [{"fold": f["fold"], "alpha": f["alpha"], "r2": f["r2"].tolist()} for f in per_fold],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
