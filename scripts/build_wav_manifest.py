#!/usr/bin/env python3
import argparse
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from tqdm import tqdm


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


def _get_num_frames(path: str) -> int:
    try:
        with wave.open(path, "rb") as w:
            return int(w.getnframes())
    except Exception:
        info = sf.info(path)
        return int(info.frames)

def _read_frames(item):
    path, relpath = item
    try:
        frames = _get_num_frames(path)
        return relpath, frames
    except Exception as exc:
        return "__ERROR__", f"{path}: {exc}"


def main():
    ap = argparse.ArgumentParser(description="Build a Fairseq wav manifest.")
    ap.add_argument("--root", required=True, help="Root directory to scan.")
    ap.add_argument("--out", required=True, help="Output manifest path.")
    ap.add_argument("--ext", default="wav", help="Audio extension (default: wav).")
    ap.add_argument("--workers", type=int, default=4, help="Number of I/O workers.")
    ap.add_argument(
        "--exclude-dirs",
        default=",".join(DEFAULT_EXCLUDE_DIRS),
        help="Comma-separated directory substrings to exclude.",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    ext = "." + args.ext.lower().lstrip(".")
    exclude_substrings = [
        s.strip().lower() for s in args.exclude_dirs.split(",") if s.strip()
    ]

    existing = set()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            existing_root = f.readline().strip()
            if existing_root and os.path.abspath(existing_root) != root:
                raise ValueError(
                    f"manifest root mismatch: {existing_root} vs {root}"
                )
            for line in f:
                relpath = line.split("\t", 1)[0].strip()
                if relpath:
                    existing.add(relpath)
        print(f"[INFO] loaded {len(existing)} existing entries")

    skipped = 0
    total_new = 0
    chunk_size = 10000
    buffer = []

    wav_iter = _iter_wav_files(root, exclude_substrings, ext)
    scan_bar = tqdm(
        wav_iter,
        desc="Scanned",
        unit="file",
        file=sys.stdout,
        disable=False,
        dynamic_ncols=True,
        position=0,
    )
    new_bar = tqdm(
        desc="New processed",
        unit="file",
        file=sys.stdout,
        disable=False,
        dynamic_ncols=True,
        position=1,
    )

    def iter_items():
        for path in scan_bar:
            relpath = os.path.relpath(path, root).replace(os.path.sep, "/")
            if relpath in existing:
                continue
            yield path, relpath

    mode = "a" if os.path.exists(args.out) else "w"
    with open(args.out, mode) as f:
        if mode == "w":
            f.write(root + "\n")

        try:
            if args.workers and args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    results = ex.map(_read_frames, iter_items(), chunksize=32)
                    for result in results:
                        new_bar.update(1)
                        if result is None:
                            skipped += 1
                            continue
                        if result[0] == "__ERROR__":
                            skipped += 1
                            continue
                        relpath, frames = result
                        buffer.append((relpath, frames))
                        if len(buffer) >= chunk_size:
                            for relpath, frames in buffer:
                                f.write(f"{relpath}\t{frames}\n")
                            total_new += len(buffer)
                            buffer.clear()
            else:
                for result in map(_read_frames, iter_items()):
                    new_bar.update(1)
                    if result is None:
                        skipped += 1
                        continue
                    if result[0] == "__ERROR__":
                        skipped += 1
                        continue
                    relpath, frames = result
                    buffer.append((relpath, frames))
                    if len(buffer) >= chunk_size:
                        for relpath, frames in buffer:
                            f.write(f"{relpath}\t{frames}\n")
                        total_new += len(buffer)
                        buffer.clear()

            if buffer:
                for relpath, frames in buffer:
                    f.write(f"{relpath}\t{frames}\n")
                total_new += len(buffer)
        finally:
            scan_bar.close()
            new_bar.close()

    print(
        f"[INFO] wrote {len(existing) + total_new} entries to {args.out} "
        f"(added {total_new}, skipped {skipped})"
    )


if __name__ == "__main__":
    main()
