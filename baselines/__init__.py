# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from .utils.fairseq_compat import apply_fairseq_compat_patches
from .wandb_logging import patch_fairseq_progress_bar

apply_fairseq_compat_patches()
patch_fairseq_progress_bar()

__all__ = [
    "models",
    "tasks",
    "data",
    "utils",
]


def __getattr__(name):
    if name in {"models", "tasks", "data", "utils"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
