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


def _join_with_ablation(args, grouped, descs):
    """Pair each descriptor's emotion-relevance (Pre-flight A) with its
    encoder-decodability (Pre-flight B).

    Neither number decides anything alone. A descriptor worth adding as a pretext
    target has to be emotion-relevant AND not already decodable from the encoder:

                        already decodable      not decodable
      carries emotion   teaches little         <- the one you want
      no emotion        drop                   drop

    Cross-referencing two files by hand is how that pairing gets skipped, so it is
    printed here whenever the ablation JSON can be found.
    """
    cand = args.ablation_json
    if cand is None:
        p = str(args.prosody_prefix)
        guesses = [f"{p}.ablation.json"]
        if p.endswith("_all"):          # B is usually pointed at the +candidates prefix
            guesses.append(f"{p[:-4]}.ablation.json")
        cand = next((g for g in guesses if Path(g).exists()), None)
    if cand is None or not Path(cand).exists():
        print("\n(no ablation JSON found -- run Pre-flight A first for the joined view)")
        return None

    abl = json.loads(Path(cand).read_text())
    acc, wa_floor = abl.get("accuracy", {}), abl.get("mean_per_fold_majority", 0.0)

    def _m(key, metric):
        """Rows are {'wa','ua','recall'} dicts; tolerate the older float format."""
        v = acc.get(key)
        if v is None:
            return None
        return v.get(metric) if isinstance(v, dict) else v

    # UA is the relevance metric, not WA. A descriptor that separates one class
    # pair -- the ang/hap valence contrast at comparable arousal is the case in
    # point -- barely moves overall accuracy while clearly moving per-class
    # recall. Judging relevance on WA would retire exactly that descriptor.
    n_cls = next((len(v["recall"]) for v in acc.values()
                  if isinstance(v, dict) and v.get("recall")), 4)
    ua_floor = 1.0 / n_cls
    print("\n--- joined: emotion relevance (A) x encoder decodability (B) ---")
    print(f"{'descriptor':<16} {'A: UA only':>11} {'A: dUA drop':>12} {'B: R2':>8}  verdict")
    rows = {}
    for d in descs:
        only_ua, only_wa = _m(f"{d} only", "ua"), _m(f"{d} only", "wa")
        base_ua = _m("all three", "ua")
        drop = (base_ua - _m(f"without {d}", "ua")) if _m(f"without {d}", "ua") is not None else None
        r2 = grouped.get(d)
        relevant = only_ua is not None and (only_ua - ua_floor) > 0.02
        novel = r2 is not None and r2 < 0.5
        verdict = ("keep/promote" if relevant and novel else
                   "already encoded" if relevant else
                   "no emotion signal" if only_ua is not None else "-")
        rows[d] = {"only_ua": only_ua, "only_wa": only_wa, "ua_drop": drop,
                   "r2": r2, "verdict": verdict}
        print(f"{d:<16} {only_ua * 100 if only_ua is not None else float('nan'):>11.2f} "
              f"{drop * 100 if drop is not None else float('nan'):>12.2f} "
              f"{r2 if r2 is not None else float('nan'):>8.3f}  {verdict}")
    print(f"A: UA only   = UA from that descriptor alone (UA floor {ua_floor * 100:.2f}%, "
          f"WA floor {wa_floor * 100:.2f}%)")
    print("A: dUA drop  = UA lost by removing it (training descriptors only)")
    print("B: R2        = how well the frozen encoder already predicts it")
    print("UA not WA, deliberately: a descriptor carrying one class pair moves UA")
    print("and barely moves WA. Per-class recall is in the ablation JSON.")
    print("Thresholds are crude (relevance >2pt over floor, novelty R2<0.5) -- read")
    print("the numbers, not the verdict column.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation_json", default=None,
                    help="Pre-flight A's <prefix>.ablation.json, for the joined view. "
                         "Auto-located next to --prosody_prefix when omitted.")
    ap.add_argument("--feat_prefix", required=True, help="Encoder features (from extract_eat_iemocap_features.py).")
    ap.add_argument("--prosody_prefix", required=True, help="Summary target (preflight_prosody_features.py --variant summary).")
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0],
                    help="Ridge penalties; the best is chosen per fold on an inner split of the training sessions.")
    ap.add_argument("--output_json", default=None, help="Where to write the numbers (default: <feat_prefix>.prosody_r2.json).")
    ap.add_argument("--label", default=None, help="Name for this encoder in the printout, e.g. 'frozen-EAT'.")
    ap.add_argument(
        "--skip_session_check",
        action="store_true",
        help="Proceed when the utterance count does not match the 5531-utterance "
             "IEMOCAP total. Folds are positional, so the result is then computed "
             "over mixed sessions and must not be recorded as the baseline.",
    )
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

    # Descriptor grouping comes from the sidecar written by Pre-flight A, so this
    # works for the training three and for the "+candidates" prefix alike rather
    # than hardcoding a width.
    descs, stats = list(DESCRIPTORS), list(STATS)
    dims_path = Path(f"{args.prosody_prefix}.dims.json")
    if dims_path.exists():
        meta = json.loads(dims_path.read_text())
        descs = meta.get("descriptors", descs)
        stats = meta.get("stats", stats)
    assert Y.shape[1] == len(descs) * len(stats), (
        f"expected {len(descs) * len(stats)} dims for descriptors {descs}, got "
        f"{Y.shape[1]}. Pass --variant summary to preflight_prosody_features.py."
    )
    # Assert rather than warn, matching preflight_prosody_features.py. The folds
    # are positional, so on a filtered subset the ridge would be fit across mixed
    # sessions -- and these numbers are what the plan says to record as the
    # frozen-EAT baseline the post-training run is compared against. A printed
    # warning is easy to lose in a tee'd log; a wrong baseline is not recoverable
    # later, because the comparison run will look like a legitimate change.
    if sum(SESSION_SIZES) != X.shape[0] and not args.skip_session_check:
        raise ValueError(
            f"{X.shape[0]} utterances != {sum(SESSION_SIZES)} expected, so the fixed "
            "positional bounds are not the LOSO sessions and the folds would mix "
            "speakers. Pass --skip_session_check to proceed anyway, and do NOT record "
            "the result as the frozen-EAT baseline."
        )

    bounds = np.cumsum([0] + SESSION_SIZES)
    n_folds = sum(1 for i in range(5) if bounds[i + 1] <= X.shape[0])

    # Rows past the last complete fold belong to sessions that are never held out.
    # Leaving them in `tr` would put those speakers in every fold's training set --
    # silently, since they are also never tested and so never flagged. Restrict the
    # whole problem to the covered prefix instead.
    covered = int(bounds[n_folds])
    if covered < X.shape[0]:
        print(f"[warn] dropping rows {covered}:{X.shape[0]} -- past the last complete "
              f"fold, so those sessions would train in every fold and test in none.")
        X, Y = X[:covered], Y[:covered]

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

    names = [f"{d}_{s}" for d in descs for s in stats]
    grouped = {
        d: float(r2_mean[j * len(stats):(j + 1) * len(stats)].mean())
        for j, d in enumerate(descs)
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

    joined = _join_with_ablation(args, grouped, descs)

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
                "joined_with_ablation": joined,
                "overall_r2": float(r2_mean.mean()),
                "per_fold": [{"fold": f["fold"], "alpha": f["alpha"], "r2": f["r2"].tolist()} for f in per_fold],
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
