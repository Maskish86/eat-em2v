# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .utils.fairseq_compat import apply_fairseq_compat_patches
from .wandb_logging import patch_fairseq_progress_bar

apply_fairseq_compat_patches()
patch_fairseq_progress_bar()

from . import models
from . import tasks
from . import data
from . import utils

__all__ = [
    "models",
    "tasks",
    "data",
    "utils",
]
