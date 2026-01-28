import atexit
import logging
import os
import uuid
from numbers import Number

try:
    import torch
except Exception:
    torch = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


logger = logging.getLogger(__name__)
_wandb_run = None


def _wandb_disabled() -> bool:
    disabled = os.environ.get("WANDB_DISABLED", "")
    if disabled.lower() in {"1", "true", "yes"}:
        return True
    mode = os.environ.get("WANDB_MODE", "")
    return mode.lower() == "disabled"


def _is_master_process() -> bool:
    if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


def _finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def _get_or_create_run_id() -> str | None:
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        return run_id
    run_id_file = os.environ.get("WANDB_RUN_ID_FILE", ".wandb_run_id")
    try:
        if os.path.exists(run_id_file):
            with open(run_id_file, "r", encoding="utf-8") as f:
                run_id = f.read().strip()
                if run_id:
                    os.environ["WANDB_RUN_ID"] = run_id
                    return run_id
        run_id = uuid.uuid4().hex
        os.makedirs(os.path.dirname(os.path.abspath(run_id_file)), exist_ok=True)
        with open(run_id_file, "w", encoding="utf-8") as f:
            f.write(run_id)
        os.environ["WANDB_RUN_ID"] = run_id
        return run_id
    except Exception as exc:
        logger.warning("Failed to persist WANDB_RUN_ID: %s", exc)
        return None


def _get_wandb_run():
    global _wandb_run
    if _wandb_run is not None:
        return _wandb_run
    if wandb is None:
        if os.environ.get("WANDB_PROJECT"):
            logger.warning("wandb not installed; set WANDB_PROJECT but logging is disabled.")
        return None
    if _wandb_disabled() or not _is_master_process():
        return None
    if wandb.run is not None:
        _wandb_run = wandb.run
        return _wandb_run
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    name = os.environ.get("WANDB_NAME")
    group = os.environ.get("WANDB_GROUP")
    job_type = os.environ.get("WANDB_JOB_TYPE")
    run_id = _get_or_create_run_id()
    resume = os.environ.get("WANDB_RESUME")
    if run_id and not resume:
        resume = "allow"
    try:
        _wandb_run = wandb.init(
            project=project,
            name=name,
            group=group,
            job_type=job_type,
            id=run_id,
            resume=resume,
        )
    except Exception as exc:
        logger.warning("wandb.init failed: %s", exc)
        _wandb_run = None
        return None
    atexit.register(_finish_wandb)
    return _wandb_run


def _format_stats(stats, tag=None):
    payload = {}
    prefix = f"{tag}/" if tag else ""
    for key, value in stats.items():
        if key == "num_updates":
            continue
        if torch is not None and torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, Number):
            payload[prefix + key] = value
    return payload


def configure_from_task_cfg(cfg) -> None:
    enabled = getattr(cfg, "wandb_enable", False)
    if not enabled:
        return
    project = getattr(cfg, "wandb_project", None)
    if project and not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_PROJECT"] = str(project)
    run_name = getattr(cfg, "wandb_run_name", None)
    if run_name and not os.environ.get("WANDB_NAME"):
        os.environ["WANDB_NAME"] = str(run_name)
    group = getattr(cfg, "wandb_group", None)
    if group and not os.environ.get("WANDB_GROUP"):
        os.environ["WANDB_GROUP"] = str(group)
    job_type = getattr(cfg, "wandb_job_type", None)
    if job_type and not os.environ.get("WANDB_JOB_TYPE"):
        os.environ["WANDB_JOB_TYPE"] = str(job_type)
    if os.environ.get("WANDB_PROJECT") is None:
        logger.warning("wandb_enable is true but WANDB_PROJECT is not set.")


class WandbProgressBarWrapper:
    def __init__(self, wrapped_bar):
        self.wrapped_bar = wrapped_bar

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        run = _get_wandb_run()
        if run is not None:
            if step is None:
                step = stats.get("num_updates")
            payload = _format_stats(stats, tag=tag)
            if payload:
                wandb.log(payload, step=step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        run = _get_wandb_run()
        if run is not None:
            if step is None:
                step = stats.get("num_updates")
            payload = _format_stats(stats, tag=tag)
            if payload:
                wandb.log(payload, step=step)
        self.wrapped_bar.print(stats, tag=tag, step=step)


def patch_fairseq_progress_bar() -> None:
    try:
        from fairseq.logging import progress_bar as fs_progress_bar
    except Exception:
        return
    if getattr(fs_progress_bar, "_wandb_patched_by_baselines", False):
        return

    original_progress_bar = fs_progress_bar.progress_bar

    def wrapped_progress_bar(*args, **kwargs):
        bar = original_progress_bar(*args, **kwargs)
        if _get_wandb_run() is None:
            return bar
        return WandbProgressBarWrapper(bar)

    fs_progress_bar.progress_bar = wrapped_progress_bar
    fs_progress_bar._wandb_patched_by_baselines = True
