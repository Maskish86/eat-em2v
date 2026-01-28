#!/usr/bin/env python3
import argparse
import os
import sys

import soundfile as sf


DEFAULT_EXCLUDE_DIRS = [
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "tmp",
    "cache",
    "logs",
]


def _should_skip_dir(name, exclude_substrings):
    lower_name = name.lower()
    return any(sub in lower_name for sub in exclude_substrings)


def _iter_wav_files(root, exclude_substrings, ext):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not _should_skip_dir(d, exclude_substrings)
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            if not fname.lower().endswith(ext):
                continue
            path = os.path.join(dirpath, fname)
            if os.path.isfile(path):
                yield path


def main():
    ap = argparse.ArgumentParser(description="Build a Fairseq wav manifest.")
    ap.add_argument("--root", required=True, help="Root directory to scan.")
    ap.add_argument("--out", required=True, help="Output manifest path.")
    ap.add_argument("--ext", default="wav", help="Audio extension (default: wav).")
    ap.add_argument(
        "--exclude-dirs",
        default=",".join(DEFAULT_EXCLUDE_DIRS),
        help="Comma-separated directory substrings to exclude.",
    )
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"[INFO] manifest already exists at {args.out}; skipping.")
        return

    root = os.path.abspath(args.root)
    ext = "." + args.ext.lower().lstrip(".")
    exclude_substrings = [
        s.strip().lower() for s in args.exclude_dirs.split(",") if s.strip()
    ]

    entries = []
    skipped = 0

    for path in _iter_wav_files(root, exclude_substrings, ext):
        try:
            info = sf.info(path)
            frames = int(info.frames)
        except Exception as exc:
            print(f"[WARN] failed to read {path}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        relpath = os.path.relpath(path, root).replace(os.path.sep, "/")
        entries.append((relpath, frames))

    entries.sort(key=lambda item: item[0])

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(root + "\n")
        for relpath, frames in entries:
            f.write(f"{relpath}\t{frames}\n")

    print(
        f"[INFO] wrote {len(entries)} entries to {args.out} (skipped {skipped})"
    )


if __name__ == "__main__":
    main()
