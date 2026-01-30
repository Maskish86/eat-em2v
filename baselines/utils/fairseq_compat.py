import inspect

import torch.nn as nn
import torch.nn.functional as F


def ensure_samepad2d():
    try:
        from fairseq import modules as fairseq_modules
    except Exception:
        return
    if hasattr(fairseq_modules, "SamePad2d"):
        return

    class SamePad2d(nn.Module):
        """
        Replacement for fairseq.modules.SamePad2d (removed in fairseq 0.12).
        Mirrors fairseq SamePad behavior by trimming extra rows/cols for even kernels.
        """

        def __init__(self, kernel_size, stride=1, dilation=1):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            self.kernel_size = kernel_size
            self.stride = stride
            self.dilation = dilation
            self.remove_h = 1 if self.kernel_size[0] % 2 == 0 else 0
            self.remove_w = 1 if self.kernel_size[1] % 2 == 0 else 0

        def forward(self, x):
            if self.remove_h:
                x = x[:, :, :-self.remove_h, :]
            if self.remove_w:
                x = x[:, :, :, :-self.remove_w]
            return x

    fairseq_modules.SamePad2d = SamePad2d


def ensure_transpose_last_kwarg():
    try:
        from fairseq import modules as fairseq_modules
    except Exception:
        return

    base_cls = getattr(fairseq_modules, "TransposeLast", None)
    if base_cls is None:
        return

    if getattr(base_cls, "_eat_transpose_last_shim", False):
        return

    class TransposeLast(nn.Module):
        _eat_transpose_last_shim = True

        def __init__(self, dims=None, *args, **kwargs):
            super().__init__()
            dim = None

            # Handle EAT typo and legacy keyword
            if "tranpose_dim" in kwargs:
                dim = kwargs.pop("tranpose_dim")
            elif "transpose_dim" in kwargs:
                dim = kwargs.pop("transpose_dim")

            if dims is None and dim is not None:
                dims = dim

            self._permute = None
            if dims is None:
                self._dims = None
            elif isinstance(dims, (list, tuple)):
                if len(dims) == 2:
                    self._dims = (dims[0], dims[1])
                else:
                    self._dims = None
                    self._permute = tuple(dims)
            else:
                self._dims = (dims, -1)

        def forward(self, x):
            if self._permute is not None:
                return x.permute(*self._permute)
            if self._dims is None:
                return x.transpose(-1, -2)
            return x.transpose(self._dims[0], self._dims[1])

    fairseq_modules.TransposeLast = TransposeLast


def ensure_ema_module_config():
    try:
        from fairseq import modules as fairseq_modules
    except Exception:
        return

    ema_cls = getattr(fairseq_modules, "EMAModuleConfig", None)
    if ema_cls is None:
        return

    if getattr(ema_cls, "_eat_ema_config_shim", False):
        return

    try:
        sig = inspect.signature(ema_cls)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            return
        allowed = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        allowed.discard("self")
        if "log_norms" in allowed and "add_missing_params" in allowed:
            return
    else:
        allowed = None

    orig_init = ema_cls.__init__

    def __init__(self, *args, **kwargs):
        if allowed is None:
            kwargs.pop("log_norms", None)
            kwargs.pop("add_missing_params", None)
            return orig_init(self, *args, **kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig_init(self, *args, **filtered_kwargs)

    ema_cls.__init__ = __init__
    ema_cls._eat_ema_config_shim = True


def ensure_ema_module():
    try:
        from fairseq import modules as fairseq_modules
    except Exception:
        return

    ema_cls = getattr(fairseq_modules, "EMAModule", None)
    if ema_cls is None:
        return

    if getattr(ema_cls, "_eat_ema_module_shim", False):
        return

    try:
        sig = inspect.signature(ema_cls.__init__)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ):
            return
        allowed = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        allowed.discard("self")
        if "copy_model" in allowed:
            return
    else:
        allowed = None

    orig_init = ema_cls.__init__
    orig_set_decay = getattr(ema_cls, "set_decay", None)

    def __init__(self, *args, **kwargs):
        if allowed is None:
            kwargs.pop("copy_model", None)
            return orig_init(self, *args, **kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig_init(self, *args, **filtered_kwargs)

    ema_cls.__init__ = __init__

    if orig_set_decay is not None:
        def set_decay(self, *args, **kwargs):
            kwargs.pop("weight_decay", None)
            return orig_set_decay(self, *args, **kwargs)
        ema_cls.set_decay = set_decay

    ema_cls._eat_ema_module_shim = True


def apply_fairseq_compat_patches():
    ensure_samepad2d()
    ensure_transpose_last_kwarg()
    ensure_ema_module_config()
    ensure_ema_module()


__all__ = [
    "apply_fairseq_compat_patches",
    "ensure_samepad2d",
    "ensure_transpose_last_kwarg",
    "ensure_ema_module_config",
    "ensure_ema_module",
]
