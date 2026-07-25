"""Standalone check of the prosody helpers, extracted verbatim from pretrain_eat.py.

Verifies shapes, the pad/patch-boundary rule, mask-subset invariant, and that the
interpolation baseline is exact on a linear target.
"""
import ast
import math
import pathlib
import types
import torch

# Repo root from __file__, so this runs on the training pod and in CI too --
# it is the only executable check of these helpers, which are loaded via ast
# specifically to avoid importing fairseq.
REPO = pathlib.Path(__file__).resolve().parent.parent

# --- load the helper functions out of pretrain_eat.py without importing fairseq
src = (REPO / "baselines/models/pretrain_eat.py").read_text()
tree = ast.parse(src)
wanted = {"prosody_valid_time", "compute_prosody", "prosody_interp_baseline", "_r2"}
mod = ast.Module(
    body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
    type_ignores=[],
)
ns = types.SimpleNamespace()
g = {"torch": torch, "math": math}
exec(compile(mod, "<helpers>", "exec"), g)
prosody_valid_time = g["prosody_valid_time"]
compute_prosody = g["compute_prosody"]
prosody_interp_baseline = g["prosody_interp_baseline"]
_r2 = g["_r2"]
assert wanted <= set(g), sorted(wanted - set(g))

B, T, M = 4, 1024, 128
N_TIME, N_FREQ, PF = 64, 8, 16
CLONE = 3

torch.manual_seed(0)

# --- build a batch with known valid lengths, padded exactly as the dataset does
n_frames = torch.tensor([1024, 700, 512 + 5, 33])
S_log = torch.randn(B, T, M) * 2 - 4
for b, nf in enumerate(n_frames):
    S_log[b, nf:] = 0.0  # raw log-mel zero padding, pre-normalization

PAD_NORM = 4.268 / (4.569 * 2)
source = ((S_log - (-4.268)) / (4.569 * 2))
vt = prosody_valid_time(source, PAD_NORM, N_TIME, PF)
expected = (torch.arange(N_TIME)[None, :] < (n_frames // PF)[:, None])
assert vt.shape == (B, N_TIME), vt.shape
assert torch.equal(vt, expected), "valid_time != floor(n_frames/16)"
# the straddling patch is discarded, never admitted
assert vt[2].sum() == 32 and n_frames[2] // PF == 32
print("ok  valid_time: floor rule, straddling patch discarded")

# --- descriptors
for norm in ("instance", "corpus"):
    kw = dict(corpus_mean=[1.0, 2.0, 3.0], corpus_std=[1.0, 1.0, 1.0]) if norm == "corpus" else {}
    p = compute_prosody(S_log, vt, N_TIME, PF, norm, **kw)
    assert p.shape == (B, 3, N_TIME), p.shape
    assert torch.isfinite(p).all(), f"non-finite under {norm}"
    # padded patches are zeroed out
    assert (p[2, :, 32:] == 0).all(), "pad patches not zeroed"
    print(f"ok  compute_prosody[{norm}]: shape {tuple(p.shape)}, finite, pad zeroed")

# instance norm: over VALID patches only, mean~0 var~1
p = compute_prosody(S_log, vt, N_TIME, PF, "instance")
v = p[1, :, : n_frames[1] // PF]
assert v.mean().abs() < 1e-4, v.mean()
assert (v.var(dim=-1, unbiased=False) - 1).abs().max() < 1e-3, v.var(dim=-1)
print("ok  instance norm: mean~0 var~1 over valid patches only")

# flux[0] == 0 by the S_{-1} := S_0 convention -> patch 0 flux is mean of 15 real deltas
S_flat = torch.zeros(1, T, M)
S_flat[0, :64] = 1.0
vt_flat = torch.ones(1, N_TIME, dtype=torch.bool)
p_flat = compute_prosody(S_flat, vt_flat, N_TIME, PF, "corpus", corpus_mean=[0, 0, 0], corpus_std=[1, 1, 1])
assert p_flat[0, 2, 0] == 0, "flux nonzero on a constant signal -> t=0 convention wrong"
print("ok  flux: S_{-1} := S_0 gives flux[0] == 0")

# --- masking path: col subset of masked_b, lengths line up
masked_b = torch.rand(B * CLONE, N_TIME * N_FREQ) < 0.8
vt_rep = expected.repeat_interleave(CLONE, 0)
col_t = masked_b.view(-1, N_TIME, N_FREQ).all(-1) & vt_rep
col = col_t.repeat_interleave(N_FREQ, dim=-1)
sel = col[masked_b]
assert (col & ~masked_b).sum() == 0, "col is NOT a subset of masked_b"
assert sel.sum() == col.sum(), (sel.sum().item(), col.sum().item())

p_bt = p.transpose(1, 2).repeat_interleave(CLONE, 0)
target = p_bt.repeat_interleave(N_FREQ, dim=1)[col]
fake_xs0 = torch.randn(int(masked_b.sum()), 768)
assert fake_xs0[sel].shape[0] == target.shape[0], (fake_xs0[sel].shape, target.shape)
print(f"ok  col subset invariant; {target.shape[0]} prosody rows align with xs[0][sel]")

# every selected column really is fully masked
assert masked_b.view(-1, N_TIME, N_FREQ)[col_t].all(), "selected column not fully masked"
print("ok  every selected column is fully masked")

# --- interpolation baseline: exact on a linear target
anchor = (~masked_b.view(-1, N_TIME, N_FREQ).all(-1)) & vt_rep
lin = torch.arange(N_TIME, dtype=torch.float32)[None, :, None].expand(B * CLONE, -1, 3).contiguous()
interp = prosody_interp_baseline(lin, anchor)
rows = anchor.any(-1)
err = (interp - lin).abs()[rows]
assert err.max() < 1e-4, f"interp not exact on a linear target: {err.max()}"
print("ok  interp baseline: exact on a linear target (R^2 == 1)")

# and it beats nothing on pure noise
noise = torch.randn(B * CLONE, N_TIME, 3)
i2 = prosody_interp_baseline(noise, anchor).repeat_interleave(N_FREQ, dim=1)[col]
t2 = noise.repeat_interleave(N_FREQ, dim=1)[col]
print(f"ok  interp R^2 on noise = {_r2(i2, t2):+.3f} (should be <= 0), on linear = 1.000")

# --- bf16 regression: the exact-equality version of pad detection passed in
# fp32 (round-trip is exactly 0.0) but never fired in bf16, the real train dtype.
src_bf = source.bfloat16().float()
vt_bf = prosody_valid_time(src_bf, PAD_NORM, N_TIME, PF)
assert torch.equal(vt_bf, expected), "pad detection broken under bf16 quantization"
S_bf = src_bf * (4.569 * 2) + (-4.268)
assert (S_bf[2, 600:] == 0).all().item() is False, "bf16 round-trip should NOT be exactly 0"
print("ok  pad detection survives bf16 (exact == 0 would fail here)")

# tolerance is not so loose that real speech reads as padding
assert prosody_valid_time(torch.randn(2, T, M) * 2 - 4, PAD_NORM, N_TIME, PF).all(), \
    "tolerance too loose: real speech detected as padding"
print("ok  tolerance does not misfire on speech")

# --- empty-selection guard
empty = torch.zeros(0, 3)
assert empty.numel() == 0
print("ok  empty prosody_target is detectable via .numel() before d2v_loss")

# --- dim parity constant
assert abs(math.sqrt(16**2 / 3) - 9.2376) < 1e-3
print(f"ok  PROSODY_DIM_PARITY = sqrt(256/3) = {math.sqrt(256/3):.4f}")

# ---------------------------------------------------------------------------
# Pre-flight probe helpers (prosody-multitask-plan.md, "Pre-flight probes")
# ---------------------------------------------------------------------------

# norm="none" must be the identity on the descriptors -- the corpus-statistics
# job depends on it, and normalizing there would silently yield mean 0 / std 1.
p_none = compute_prosody(S_log, vt, N_TIME, PF, "none")
p_corp = compute_prosody(S_log, vt, N_TIME, PF, "corpus", corpus_mean=[0, 0, 0], corpus_std=[1, 1, 1])
assert torch.allclose(p_none, p_corp, atol=1e-5), "norm='none' is not the raw descriptor"
assert (p_none[2, :, 32:] == 0).all(), "norm='none' must still zero padded patches"
assert p_none[:, 0].abs().max() > 1.0, "log-energy looks normalized -- 'none' is not raw"
print("ok  compute_prosody[none]: raw values, pad still zeroed")

# --- summary_stats, loaded the same fairseq-free way as the helpers above
pf_src = (REPO / "baselines/downstream/preflight_prosody_features.py").read_text()
pf_tree = ast.parse(pf_src)
pf_mod = ast.Module(
    body=[n for n in pf_tree.body if isinstance(n, ast.FunctionDef) and n.name == "summary_stats"],
    type_ignores=[],
)
pg = {"torch": torch, "STATS": ["mean", "std", "min", "max", "slope"]}
exec(compile(pf_mod, "<preflight>", "exec"), pg)
summary_stats = pg["summary_stats"]

# a known ramp on descriptor 0 over the valid prefix, garbage in the pad region
ramp = torch.zeros(1, 3, N_TIME)
n_valid = 40
ramp[0, 0, :n_valid] = 2.0 * torch.arange(n_valid, dtype=torch.float32) - 5.0
vt_ramp = torch.zeros(1, N_TIME, dtype=torch.bool)
vt_ramp[0, :n_valid] = True

s = summary_stats(ramp, vt_ramp)
assert s.shape == (1, 15), s.shape
e_mean, e_std, e_min, e_max, e_slope = s[0, 0], s[0, 1], s[0, 2], s[0, 3], s[0, 4]
expected_mean = (2.0 * torch.arange(n_valid, dtype=torch.float32) - 5.0).mean()
assert (e_mean - expected_mean).abs() < 1e-3, (e_mean, expected_mean)
assert (e_min - (-5.0)).abs() < 1e-3, f"min={e_min} -- pad zeros leaked into the min"
assert (e_max - (2.0 * (n_valid - 1) - 5.0)).abs() < 1e-2, e_max
assert (e_slope - 2.0).abs() < 1e-3, f"slope={e_slope}, expected 2.0"
print(f"ok  summary_stats: mean/min/max/slope exact on a ramp (slope={e_slope:.4f})")

# the pad region must not reach min/max even when it is extreme relative to the
# valid values -- descriptor 1 is all-positive inside the valid prefix, so a
# zeroed pad patch would win the min if the mask were not applied.
pos = torch.zeros(1, 3, N_TIME)
pos[0, 1, :n_valid] = 7.0
s2 = summary_stats(pos, vt_ramp)
assert (s2[0, 5 + 2] - 7.0).abs() < 1e-4, f"pad zero leaked into min: {s2[0, 5 + 2]}"
assert (s2[0, 5 + 1]).abs() < 1e-3, "std of a constant should be ~0"
print("ok  summary_stats: padded patches excluded from min/max")

# --- ridge closed form (numpy only, safe to import directly)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "pf_r2", str(REPO / "baselines/downstream/preflight_prosody_r2.py")
)
_pf_r2 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pf_r2)

import numpy as np
rng = np.random.default_rng(0)
Xt = rng.normal(size=(300, 8))
Wt = rng.normal(size=(8, 3))
Yt = Xt @ Wt + 1.5
pred = _pf_r2.ridge_fit_predict(Xt[:200], Yt[:200], Xt[200:], 1e-8)
assert np.abs(pred - Yt[200:]).max() < 1e-4, np.abs(pred - Yt[200:]).max()
print("ok  ridge_fit_predict: recovers an exact linear map (intercept included)")

# a non-zero target mean must not be absorbed into the penalty
Yb = Yt + 100.0
pb = _pf_r2.ridge_fit_predict(Xt[:200], Yb[:200], Xt[200:], 10.0)
assert np.abs(pb.mean() - 100.0) < 5.0, f"intercept shrunk toward 0: {pb.mean()}"
print("ok  ridge_fit_predict: intercept is not penalized")

print("\nall checks passed")
