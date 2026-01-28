#!/usr/bin/env python3
"""
Evaluate frozen EAT/EAT-em2v representations on IEMOCAP using the
emotion2vec downstream protocol (linear head on cached features).

This mirrors external/emotion2vec/iemocap_downstream/main.py but adds:
- Argparse interface (feature prefix, seed, output_dir, W&B)
- Optional W&B logging for provenance/metrics per fold
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

import wandb

from external.emotion2vec.iemocap_downstream import data as emo_data
from external.emotion2vec.iemocap_downstream import model as emo_model
from external.emotion2vec.iemocap_downstream import utils as emo_utils


LABEL_DICT = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}
SESSION_SIZES = [1085, 1023, 1151, 1031, 1241]  # Session1-5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_splits():
    offsets = np.cumsum([0] + SESSION_SIZES)
    splits = []
    for i in range(5):
        start, end = offsets[i], offsets[i + 1]
        test_idx = (start, end)
        splits.append(test_idx)
    return splits


def train_one_epoch(model, optimizer, criterion, train_loader, device, scheduler=None):
    """
    Per-batch scheduler stepping to follow CyclicLR design intent.
    """
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats, padding_mask)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item()

    return train_loss


def train_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve feature files from prefix and validate presence
    feat_prefix = Path(args.feat_prefix)
    for suffix in (".npy", ".lengths", ".emo"):
        if not feat_prefix.with_suffix(suffix).exists():
            raise FileNotFoundError(f"Missing feature file: {feat_prefix.with_suffix(suffix)}")

    iemocap_data = emo_data.load_ssl_features(str(feat_prefix), LABEL_DICT, max_speech_seq_len=None)
    input_dim = int(iemocap_data["feats"].shape[1])

    splits = get_splits()
    wa_sum = ua_sum = f1_sum = 0.0

    wb_run = None
    if args.wandb_project:
        try:
            wb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or "iemocap-downstream",
                group=args.wandb_group or None,
                job_type="downstream_eval",
                config=vars(args),
            )
        except wandb.errors.CommError as exc:
            print(f"W&B init failed ({exc}); continuing without W&B logging.")
            wb_run = None

    for fold, (test_start, test_end) in enumerate(splits, 1):
        train_loader, val_loader, test_loader = emo_data.train_valid_test_iemocap_dataloader(
            iemocap_data,
            batch_size=args.batch_size,
            test_start=test_start,
            test_end=test_end,
            eval_is_test=args.eval_is_test,
        )

        model = emo_model.BaseModel(input_dim=input_dim, output_dim=len(LABEL_DICT)).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args.lr, max_lr=args.max_lr, step_size_up=args.step_size_up
        )
        criterion = nn.CrossEntropyLoss()

        best_val_wa = -1
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, scheduler=scheduler)

            val_wa, val_ua, val_f1 = emo_utils.validate_and_test(
                model, val_loader, device, num_classes=len(LABEL_DICT)
            )

            if val_wa > best_val_wa:
                best_val_wa = val_wa
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            if wb_run:
                wandb.log(
                    {
                        f"fold_{fold}/train_loss": train_loss / len(train_loader),
                        f"fold_{fold}/val_WA": val_wa,
                        f"fold_{fold}/val_UA": val_ua,
                        f"fold_{fold}/val_F1": val_f1,
                        "epoch": epoch + 1,
                        "fold": fold,
                    }
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        test_wa, test_ua, test_f1 = emo_utils.validate_and_test(
            model, test_loader, device, num_classes=len(LABEL_DICT)
        )

        wa_sum += test_wa
        ua_sum += test_ua
        f1_sum += test_f1

        ckpt_path = output_dir / f"iemocap_fold{fold}.pth"
        torch.save(model.state_dict(), ckpt_path)

        if wb_run:
            wandb.log(
                {
                    f"fold_{fold}/test_WA": test_wa,
                    f"fold_{fold}/test_UA": test_ua,
                    f"fold_{fold}/test_F1": test_f1,
                    "fold": fold,
                }
            )
            artifact = wandb.Artifact(
                name=f"iemocap-fold{fold}-checkpoint",
                type="model",
                metadata={"fold": fold},
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)

    avg_wa = wa_sum / len(splits)
    avg_ua = ua_sum / len(splits)
    avg_f1 = f1_sum / len(splits)

    print(f"Average WA: {avg_wa:.2f}%; UA: {avg_ua:.2f}%; F1: {avg_f1:.2f}%")
    if wb_run:
        wandb.summary["avg_WA"] = avg_wa
        wandb.summary["avg_UA"] = avg_ua
        wandb.summary["avg_F1"] = avg_f1
        wandb.finish()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_prefix", required=True, help="Prefix to features (expects <prefix>.npy/.lengths/.emo)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_lr", type=float, default=1e-3)
    ap.add_argument("--step_size_up", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--eval_is_test", action="store_true", help="Use test set as validation (no val split).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="runs/iemocap_eval")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_eval(args)
