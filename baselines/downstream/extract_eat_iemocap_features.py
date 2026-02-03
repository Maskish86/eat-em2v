#!/usr/bin/env python3
"""
Extract EAT/EAT-em2v frame-level features for IEMOCAP into the emotion2vec
downstream format (train.npy, train.lengths, train.emo). Spectrograms are the
input; the encoder is only run once offline, mirroring the emotion2vec
evaluation protocol.
"""

import argparse
import math
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

import fairseq
from fairseq import utils as fairseq_utils
from omegaconf import OmegaConf, open_dict
import wandb

from baselines.utils.fairseq_compat import apply_fairseq_compat_patches

LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3}


def load_manifest_labels(manifest_path: str, label_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Manifest: first line is root, subsequent lines are <relpath>\t...
    Labels: lines are "<utt_id> <label>".
    """
    with open(manifest_path) as mf:
        root = mf.readline().strip()
        rels = [line.split("\t")[0] for line in mf if line.strip()]
    utt_ids, labels = [], []
    with open(label_path) as lf:
        for line in lf:
            parts = line.strip().split()
            if not parts:
                continue
            utt_ids.append(parts[0])
            labels.append(parts[1])
    assert len(rels) == len(labels), "Manifest/label length mismatch"
    paths = [os.path.join(root, rel) for rel in rels]
    return paths, utt_ids, labels


def load_iemocap_root(root_dir: str) -> Tuple[List[str], List[str], List[str]]:
    allowed = {"ang", "hap", "neu", "sad", "exc"}
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP root not found: {root}")

    paths, utt_ids, labels = [], [], []
    for session in range(1, 6):
        eval_dir = root / f"Session{session}" / "dialog" / "EmoEvaluation"
        if not eval_dir.exists():
            raise FileNotFoundError(f"Missing EmoEvaluation directory: {eval_dir}")
        for eval_path in sorted(eval_dir.glob("*.txt")):
            with open(eval_path, encoding="utf-8", errors="ignore") as ef:
                for line in ef:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id = None
                    label = None
                    if "\t" in line:
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            utt_id = parts[1].strip()
                            label = parts[2].strip()
                    if utt_id is None or label is None:
                        m = re.search(r"\b(Ses\d+[MF]_[^\s]+)\s+([a-z]{3})\b", line)
                        if m:
                            utt_id, label = m.group(1), m.group(2)
                    if utt_id is None or label is None:
                        continue
                    if label not in allowed:
                        continue
                    if label == "exc":
                        label = "hap"

                    session_id = utt_id[4]
                    folder = utt_id.rsplit("_", 1)[0]
                    wav_path = (
                        root
                        / f"Session{session_id}"
                        / "sentences"
                        / "wav"
                        / folder
                        / f"{utt_id}.wav"
                    )
                    if not wav_path.exists():
                        raise FileNotFoundError(f"Missing wav for {utt_id}: {wav_path}")
                    paths.append(str(wav_path))
                    utt_ids.append(utt_id)
                    labels.append(label)

    if not paths:
        raise ValueError(f"No labeled utterances found under {root}")
    return paths, utt_ids, labels


class IemocapSpecDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        utt_ids: List[str],
        labels: List[str],
        target_length: int = 1024,
        n_mels: int = 128,
        patch_size: int = 16,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        self.paths = paths
        self.utt_ids = utt_ids
        self.labels = labels
        self.target_length = target_length
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        utt = self.utt_ids[idx]
        label = self.labels[idx]

        wav, sr = sf.read(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(torch.tensor(wav).float(), sr, 16000).numpy()
        wav = torch.tensor(wav).float()
        wav = wav - wav.mean()

        mel = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.n_mels,
            dither=0.0,
        ).unsqueeze(0)  # 1 x T x n_mels

        orig_frames = mel.shape[1]
        tgt = self.target_length
        if orig_frames < tgt:
            mel = F.pad(mel, (0, 0, 0, tgt - orig_frames))
        elif orig_frames > tgt:
            start = (orig_frames - tgt) // 2
            mel = mel[:, start : start + tgt, :]

        mel = (mel - self.norm_mean) / (self.norm_std * 2)

        valid_time_blocks = min(math.ceil(orig_frames / self.patch_size), tgt // self.patch_size)

        return mel, valid_time_blocks, utt, label

    @staticmethod
    def collate_fn(batch):
        mels, valid_blocks, utts, labels = zip(*batch)
        mels = torch.stack(mels, dim=0)  # B x 1 x T x F
        valid_blocks = torch.tensor(valid_blocks, dtype=torch.long)
        return mels, valid_blocks, utts, labels


def _interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" not in checkpoint_model:
        return
    if not hasattr(model, "patch_embed") or not hasattr(model, "pos_embed"):
        return
    if not hasattr(model.patch_embed, "num_patches"):
        return

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens,
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def _load_eat_pretrained(checkpoint, target_length=None):
    state = torch.load(checkpoint, map_location="cpu")
    if state.get("args") is not None:
        # Upgrade legacy fairseq checkpoints that carry args (0.10.x style).
        state = fairseq.checkpoint_utils._upgrade_state_dict(state)
    pretrained_args = state.get("cfg") or state.get("args")
    if pretrained_args is None:
        raise ValueError("Checkpoint missing cfg/args; use backbone_type 'eat_em2v' for non-fairseq .pt files.")
    if isinstance(pretrained_args, dict):
        pretrained_args = OmegaConf.create(pretrained_args)

    if hasattr(pretrained_args, "criterion"):
        pretrained_args.criterion = None
    if hasattr(pretrained_args, "lr_scheduler"):
        pretrained_args.lr_scheduler = None

    def _set_skip_ema(cfg):
        if cfg is None:
            return
        if OmegaConf.is_config(cfg):
            with open_dict(cfg):
                cfg["skip_ema"] = True
            return
        if isinstance(cfg, dict):
            cfg["skip_ema"] = True
            return
        try:
            setattr(cfg, "skip_ema", True)
        except Exception:
            return

    model_cfg = getattr(pretrained_args, "model", None)
    try:
        has_modalities = model_cfg is not None and "modalities" in model_cfg
    except TypeError:
        has_modalities = False
    if has_modalities:
        if OmegaConf.is_config(pretrained_args):
            with open_dict(pretrained_args):
                if target_length is not None:
                    pretrained_args.model["modalities"]["image"]["target_length"] = target_length
                if "mae_masking" in pretrained_args.model["modalities"]["image"]:
                    del pretrained_args.model["modalities"]["image"]["mae_masking"]
        else:
            if target_length is not None:
                model_cfg["modalities"]["image"]["target_length"] = target_length
            if "mae_masking" in model_cfg["modalities"]["image"]:
                del model_cfg["modalities"]["image"]["mae_masking"]

    _set_skip_ema(model_cfg)
    if not OmegaConf.is_config(pretrained_args):
        _set_skip_ema(pretrained_args)

    if OmegaConf.is_config(pretrained_args):
        task = fairseq.tasks.setup_task(pretrained_args.task)
        build_cfg = pretrained_args.model
    else:
        task = fairseq.tasks.setup_task(pretrained_args)
        build_cfg = pretrained_args
    try:
        model = task.build_model(build_cfg, from_checkpoint=True)
    except TypeError:
        model = task.build_model(build_cfg)

    checkpoint_model = state.get("model", state)
    _interpolate_pos_embed(model, checkpoint_model)

    if "modality_encoders.IMAGE.positional_encoder.pos_embed" in checkpoint_model:
        checkpoint_model["modality_encoders.IMAGE.positional_encoder.positions"] = checkpoint_model[
            "modality_encoders.IMAGE.positional_encoder.pos_embed"
        ]
        del checkpoint_model["modality_encoders.IMAGE.positional_encoder.pos_embed"]
    if "modality_encoders.IMAGE.encoder_mask" in checkpoint_model:
        del checkpoint_model["modality_encoders.IMAGE.encoder_mask"]

    model.load_state_dict(checkpoint_model, strict=True)
    return model


def _resolve_em2v_cfg(state, cfg_path=None):
    if isinstance(state, dict):
        cfg = state.get("config", None)
        if isinstance(cfg, dict) and "fairseq" in cfg:
            return cfg["fairseq"]
        if "cfg" in state:
            return state["cfg"]
        if "args" in state:
            return state["args"]
    if cfg_path:
        return OmegaConf.load(cfg_path)
    return None


def _get_cfg_node(cfg, key):
    if OmegaConf.is_config(cfg):
        return cfg[key] if key in cfg else getattr(cfg, key, None)
    if isinstance(cfg, dict):
        return cfg.get(key, None)
    return getattr(cfg, key, None)


def _select_state_dict(state):
    if not isinstance(state, dict):
        raise KeyError("Checkpoint payload is not a dict; cannot locate model weights.")
    if "student_state_dict" in state:
        return state["student_state_dict"]
    for key in ("model", "state_dict"):
        if key in state:
            return state[key]
    if all(torch.is_tensor(v) for v in state.values()):
        return state
    keys = ", ".join(sorted(state.keys()))
    raise KeyError(f"Could not find model weights in checkpoint. Keys: {keys}")


def _clear_fairseq_task(name: str, class_name: str | None = None) -> None:
    try:
        import fairseq.tasks as fairseq_tasks
    except Exception:
        return
    for registry_name in ("TASK_REGISTRY", "TASK_DATACLASS_REGISTRY"):
        registry = getattr(fairseq_tasks, registry_name, None)
        if isinstance(registry, dict):
            registry.pop(name, None)
    class_names = getattr(fairseq_tasks, "TASK_CLASS_NAMES", None)
    if isinstance(class_names, set):
        class_names.discard(name)
        if class_name:
            class_names.discard(class_name)


def _clear_fairseq_model(name: str, class_name: str | None = None) -> None:
    try:
        import fairseq.models as fairseq_models
    except Exception:
        return
    for registry_name in (
        "MODEL_REGISTRY",
        "MODEL_DATACLASS_REGISTRY",
        "ARCH_MODEL_REGISTRY",
        "ARCH_CONFIG_REGISTRY",
        "ARCH_MODEL_NAME_REGISTRY",
        "ARCH_MODEL_INV_REGISTRY",
    ):
        registry = getattr(fairseq_models, registry_name, None)
        if isinstance(registry, dict):
            registry.pop(name, None)
    class_names = getattr(fairseq_models, "MODEL_CLASS_NAMES", None)
    if isinstance(class_names, set):
        class_names.discard(name)
        if class_name:
            class_names.discard(class_name)


def _resolve_user_dir(*parts: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root.joinpath(*parts)).resolve())


def _import_user_dir(backbone_type: str) -> None:
    if backbone_type == "eat_original":
        user_dir = _resolve_user_dir("external", "EAT")
    elif backbone_type == "eat_em2v":
        user_dir = _resolve_user_dir("baselines")
    else:
        return
    fairseq_utils.import_user_module(argparse.Namespace(user_dir=user_dir))


def _ensure_eat_registrations(backbone_type: str) -> None:
    try:
        import fairseq.tasks as fairseq_tasks
        import fairseq.models as fairseq_models
    except Exception:
        return

    task_registry = getattr(fairseq_tasks, "TASK_REGISTRY", {})
    model_registry = getattr(fairseq_models, "MODEL_REGISTRY", {})

    if "mae_image_pretraining" not in task_registry:
        if backbone_type == "eat_em2v":
            import baselines.tasks.pretraining_em2v_style  # noqa: F401
        else:
            import external.EAT.tasks.pretraining_AS2M  # noqa: F401

    if "data2vec_multi" not in model_registry:
        if backbone_type == "eat_em2v":
            import baselines.models.pretrain_eat  # noqa: F401
        else:
            import external.EAT.models.EAT_pretraining  # noqa: F401


def load_eat(checkpoint, device, backbone_type="eat_original", freeze_cnn=False, target_length=None, em2v_cfg=None):
    # Register EAT modules so custom tasks/models are available.
    apply_fairseq_compat_patches()
    _clear_fairseq_task("mae_image_pretraining", "MaeImagePretrainingTask")
    _clear_fairseq_model("data2vec_multi", "Data2VecMultiModel")
    _import_user_dir(backbone_type)
    _ensure_eat_registrations(backbone_type)

    if backbone_type == "eat_original":
        model = _load_eat_pretrained(checkpoint, target_length=target_length)
    elif backbone_type == "eat_em2v":
        state = torch.load(checkpoint, map_location="cpu")
        eat_cfg = _resolve_em2v_cfg(state, em2v_cfg)
        if eat_cfg is None:
            keys = ", ".join(sorted(state.keys())) if isinstance(state, dict) else type(state).__name__
            raise KeyError(
                "EAT-em2v checkpoint is missing config. Expected state['config']['fairseq'] "
                "or state['cfg']/state['args']. Pass --em2v_cfg to load a YAML config. "
                f"Checkpoint keys: {keys}"
            )
        if not OmegaConf.is_config(eat_cfg):
            if hasattr(eat_cfg, "__dict__"):
                eat_cfg = OmegaConf.create(vars(eat_cfg))
            else:
                eat_cfg = OmegaConf.create(eat_cfg)
        raw_task_cfg = _get_cfg_node(eat_cfg, "task")
        raw_model_cfg = _get_cfg_node(eat_cfg, "model")
        if raw_task_cfg is None or raw_model_cfg is None:
            raise KeyError("EAT-em2v config must include both 'task' and 'model' sections.")
        def _project_task_cfg(raw_cfg):
            data = OmegaConf.to_container(raw_cfg, resolve=True)
            forbidden_prefixes = ("wandb_",)
            clean = {k: v for k, v in data.items() if not k.startswith(forbidden_prefixes)}
            return OmegaConf.create(clean)

        def _project_model_cfg(raw_cfg):
            data = OmegaConf.to_container(raw_cfg, resolve=True)
            forbidden = {
                "layer_decay",
                "no_decay_blocks",
                "ema_decay",
                "ema_fp32",
                "ema_start_update",
                "ema_update_freq",
            }
            clean = {k: v for k, v in data.items() if k not in forbidden}
            return OmegaConf.create(clean)

        task_cfg = _project_task_cfg(raw_task_cfg)
        model_cfg = _project_model_cfg(raw_model_cfg)
        if OmegaConf.is_config(model_cfg):
            with open_dict(model_cfg):
                model_cfg["skip_ema"] = True
        task = fairseq.tasks.setup_task(task_cfg)
        model = fairseq.models.build_model(model_cfg, task)
        model.load_state_dict(_select_state_dict(state), strict=True)
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    model.eval()
    model.to(device)

    if hasattr(model, "remove_pretraining_modules"):
        model.remove_pretraining_modules(modality="image", keep_decoder=False)

    if freeze_cnn:
        enc = getattr(model, "modality_encoders", {}).get("IMAGE", None)
        if enc is not None:
            for mod in ["local_encoder", "project_features"]:
                if hasattr(enc, mod):
                    for p in getattr(enc, mod).parameters():
                        p.requires_grad = False
    return model


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="Path to IEMOCAP manifest (tsv with root on first line).")
    ap.add_argument("--labels", help="Path to labels file (lines: <utt_id> <label>).")
    ap.add_argument("--iemocap_root", help="IEMOCAP root with Session1..Session5 (sentences/wav + dialog/EmoEvaluation).")
    ap.add_argument("--checkpoint", required=True, help="Path to EAT checkpoint.")
    ap.add_argument(
        "--backbone_type",
        choices=["eat_original", "eat_em2v"],
        default="eat_original",
        help="Backbone checkpoint type to load (eat_original follows external/EAT Fairseq checkpoint format).",
    )
    ap.add_argument(
        "--em2v_cfg",
        default=None,
        help="Optional YAML config for EAT-em2v checkpoints that lack embedded config.",
    )
    ap.add_argument("--output_prefix", required=True, help="Prefix for output files (writes .npy, .lengths, .emo).")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--norm_mean", type=float, default=-4.268)
    ap.add_argument("--norm_std", type=float, default=4.569)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb_project", default=None, help="If set, log feature extraction metadata + artifacts to W&B.")
    ap.add_argument("--wandb_name", default=None, help="Optional W&B run name.")
    ap.add_argument("--wandb_group", default=None, help="Optional W&B group name.")
    ap.add_argument("--wandb_job_type", default="feature_extraction")
    ap.add_argument("--seed", type=int, default=0, help="Seed for reproducible feature extraction ordering.")
    args = ap.parse_args()

    # Global seeding for deterministic ordering (subject to CUDA kernel determinism).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.iemocap_root:
        paths, utt_ids, labels = load_iemocap_root(args.iemocap_root)
    else:
        if not args.manifest or not args.labels:
            raise ValueError("Provide --iemocap_root or both --manifest and --labels.")
        paths, utt_ids, labels = load_manifest_labels(args.manifest, args.labels)
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
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )

    device = torch.device(args.device)
    model = load_eat(
        args.checkpoint,
        device,
        backbone_type=args.backbone_type,
        freeze_cnn=False,
        target_length=args.target_length,
        em2v_cfg=args.em2v_cfg,
    )

    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            job_type=args.wandb_job_type,
            name=args.wandb_name or f"{args.backbone_type}-iemocap",
            group=args.wandb_group or None,
            config={
                "backbone_type": args.backbone_type,
                "checkpoint": args.checkpoint,
                "target_length": args.target_length,
                "n_mels": args.n_mels,
                "patch_size": args.patch_size,
            },
        )

    freq_blocks = args.n_mels // args.patch_size
    total_tokens = (args.target_length // args.patch_size) * freq_blocks
    features = []
    lengths = []
    emo_entries = []

    for mels, valid_blocks, utts, lbls in loader:
        mels = mels.to(device)
        out = model.extract_features(mels, mask=False, remove_extra_tokens=False)
        seq = out["x"] if isinstance(out, dict) else out

        num_extra = seq.size(1) - total_tokens
        patch_tokens = seq[:, num_extra:, :]  # drop CLS/extra tokens

        for i in range(patch_tokens.size(0)):
            valid = int(valid_blocks[i].item()) * freq_blocks
            feat = patch_tokens[i, :valid].cpu().float().numpy()
            features.append(feat)
            lengths.append(feat.shape[0])
            emo_entries.append(f"{utts[i]} {lbls[i]}")

    feat_dim = features[0].shape[1] if features else 0
    total_len = sum(lengths)
    feat_mat = np.zeros((total_len, feat_dim), dtype=np.float32)
    offset = 0
    for feat in features:
        next_offset = offset + feat.shape[0]
        feat_mat[offset:next_offset] = feat
        offset = next_offset

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{prefix}.npy", feat_mat)
    with open(f"{prefix}.lengths", "w") as f_len:
        for l in lengths:
            f_len.write(f"{l}\n")
    with open(f"{prefix}.emo", "w") as f_lbl:
        for entry in emo_entries:
            f_lbl.write(entry + "\n")

    print(f"Wrote features to {prefix}.npy with {len(lengths)} utterances; feature dim={feat_dim}.")

    if wandb_run:
        wandb.log(
            {
                "num_utterances": len(lengths),
                "total_frames": sum(lengths),
                "feature_dim": feat_dim,
            }
        )
        artifact = wandb.Artifact(
            name=f"iemocap-{args.backbone_type}-features",
            type="dataset",
            metadata={
                "encoder": args.backbone_type,
                "checkpoint": os.path.basename(args.checkpoint),
            },
        )
        artifact.add_file(f"{prefix}.npy")
        artifact.add_file(f"{prefix}.lengths")
        artifact.add_file(f"{prefix}.emo")
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
