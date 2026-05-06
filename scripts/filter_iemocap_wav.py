#!/usr/bin/env python3
# Usage: python scripts/filter_iemocap_wav.py --root /path/to/IEMOCAP_full_release --out datasets/iemocap_filtered
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

VALID_EMOTIONS = {"ang", "exc", "hap", "neu", "sad"}


def iter_labels(root: Path):
    """Yield (fname, label) from raw IEMOCAP EmoEvaluation txt files."""
    for session in range(1, 6):
        eval_dir = root / f"Session{session}" / "dialog" / "EmoEvaluation"
        for txt in sorted(eval_dir.glob("*.txt")):
            with open(txt, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Ses" not in line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    fname = parts[1].strip()
                    label = parts[2].strip()
                    if label not in VALID_EMOTIONS:
                        continue
                    if label == "exc":
                        label = "hap"
                    yield fname, label


def main(args):
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root.parent / "IEMOCAP_full_release_filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(iter_labels(root))
    print(f"Found {len(entries)} utterances")

    skipped = 0
    for fname, label in tqdm(entries, unit="file"):
        session = fname[4]
        folder = fname.rsplit("_", 1)[0]
        src = root / f"Session{session}" / "sentences/wav" / folder / (fname + ".wav")
        if not src.exists():
            skipped += 1
            continue
        dst = out_dir / f"Session{session}" / folder / (fname + ".wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"Done. total={len(entries)}, skipped(missing/excluded)={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root directory of IEMOCAP_full_release")
    parser.add_argument("--out", default=None, help="output directory for filtered wav files (default: <root>_filtered)")
    main(parser.parse_args())
