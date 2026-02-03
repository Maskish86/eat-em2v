#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import subprocess


def find_checkpointN_checkpoints(ckpt_dir, every=5):
    pat = re.compile(r"checkpoint(\d+)\.pt$")
    ckpts = []
    for p in Path(ckpt_dir).glob("checkpoint*.pt"):
        m = pat.search(p.name)
        if m and int(m.group(1)) % every == 0:
            ckpts.append((int(m.group(1)), p))
    return sorted(ckpts)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    default_ckpt_dir = repo_root / "checkpoints" / "ser_pretrained"
    default_out_root = repo_root / "runs" / "iemocap_epoch_sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(default_ckpt_dir))
    ap.add_argument("--out_root", default=str(default_out_root))
    ap.add_argument("--iemocap_root", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--backbone_type", default="eat_em2v")
    ap.add_argument("--em2v_cfg", default=None)
    ap.add_argument("--wandb_project", default="eat-em2v-iemocap")
    ap.add_argument("--wandb_group", default="epoch_sweep")
    args = ap.parse_args()

    CKPT_DIR = Path(args.ckpt_dir)
    OUT_ROOT = Path(args.out_root)
    ckpts = find_checkpointN_checkpoints(CKPT_DIR, every=args.every)

    for epoch, ckpt in ckpts:
        tag = f"epoch_{epoch:03d}"
        feat_prefix = OUT_ROOT / "features" / tag

        # 1) Feature extraction
        feat_cmd = [
            "python",
            str(repo_root / "baselines" / "downstream" / "extract_eat_iemocap_features.py"),
            "--iemocap_root",
            args.iemocap_root,
        ]
        feat_cmd += [
            "--checkpoint",
            str(ckpt),
            "--backbone_type",
            args.backbone_type,
            "--output_prefix",
            str(feat_prefix),
        ]
        if args.em2v_cfg:
            feat_cmd += ["--em2v_cfg", args.em2v_cfg]
        subprocess.run(
            feat_cmd,
            check=True,
        )

        # 2) Downstream eval
        subprocess.run(
            [
                "python",
                str(repo_root / "baselines" / "downstream" / "eval_downstream_iemocap.py"),
                "--feat_prefix",
                str(feat_prefix),
                "--output_dir",
                str(OUT_ROOT / "eval" / tag),
                "--wandb_project",
                args.wandb_project,
                "--wandb_group",
                args.wandb_group,
                "--wandb_name",
                f"eval-{tag}",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
