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
        Implements TensorFlow-style 'same' padding for Conv2d.
        """

        def __init__(self, kernel_size, stride=1, dilation=1):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            self.kernel_size = kernel_size
            self.stride = stride
            self.dilation = dilation

        def forward(self, x):
            ih, iw = x.size()[-2:]
            kh, kw = self.kernel_size

            oh = (ih + self.stride - 1) // self.stride
            ow = (iw + self.stride - 1) // self.stride

            pad_h = max((oh - 1) * self.stride + (kh - 1) * self.dilation + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride + (kw - 1) * self.dilation + 1 - iw, 0)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

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

    try:
        sig = inspect.signature(base_cls.__init__)
    except (TypeError, ValueError):
        sig = None

    class TransposeLast(base_cls):
        _eat_transpose_last_shim = True

        def __init__(self, *args, **kwargs):
            dim = None

            # Handle EAT typo and legacy keyword
            if "tranpose_dim" in kwargs:
                dim = kwargs.pop("tranpose_dim")
            elif "transpose_dim" in kwargs:
                dim = kwargs.pop("transpose_dim")

            # fairseq >= 0.12: positional `dims`
            if dim is not None and sig and "dims" in sig.parameters:
                args = (dim,) if not isinstance(dim, (list, tuple)) else tuple(dim)

            super().__init__(*args, **kwargs)

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

    def __init__(self, *args, **kwargs):
        if allowed is None:
            kwargs.pop("copy_model", None)
            return orig_init(self, *args, **kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig_init(self, *args, **filtered_kwargs)

    ema_cls.__init__ = __init__
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
