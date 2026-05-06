import pandas as pd
import soundfile as sf
from pathlib import Path
import shutil
from tqdm import tqdm

tqdm.pandas()

# =========================
# Configuration
# =========================
ROOT = Path("MSP-PODCAST-Publish-2.0")
AUDIO_DIR = ROOT / "Audios"
LABEL_FILE = ROOT / "Labels" / "labels_consensus.csv"

OUT_AUDIO_DIR = ROOT / "Audios_SER_filtered"
OUT_CSV = ROOT / "Labels" / "msp_ser_filtered.csv"

MIN_SEC = 0.5
MAX_SEC = 20.0

# Standard SER emotion set
VALID_EMOTIONS = {"A", "S", "H", "U", "F", "D", "C", "N"}

# =========================
# Load consensus labels
# =========================
df = pd.read_csv(LABEL_FILE)

print(f"Loaded {len(df):,} consensus-labeled entries")

# =========================
# Label-based filtering
# =========================

# 1) Primary emotion must be standard SER
df = df[df["EmoClass"].isin(VALID_EMOTIONS)]

# 2) Require dimensional annotations
df = df[
    df["EmoAct"].notna() &
    df["EmoVal"].notna() &
    df["EmoDom"].notna()
]

# 3) Exclude hidden-label partition
df = df[df["Split_Set"] != "Test3"]

print(f"After SER label filtering: {len(df):,}")

# =========================
# Duration filtering
# =========================
def get_duration(fname: str) -> float | None:
    path = AUDIO_DIR / fname
    if not path.exists():
        return None
    try:
        with sf.SoundFile(path) as f:
            return len(f) / f.samplerate
    except Exception:
        return None

df["duration_sec"] = df["FileName"].progress_apply(get_duration)
df = df.dropna(subset=["duration_sec"])
df = df[(df["duration_sec"] >= MIN_SEC) & (df["duration_sec"] <= MAX_SEC)].reset_index(drop=True)

print(f"After duration filtering: {len(df):,}")
print(f"Total hours: {df.duration_sec.sum() / 3600:.2f}")
print(f"Average duration: {df.duration_sec.mean():.2f} s")

# =========================
# Random sampling to ~205.4 h
# =========================
TARGET_HOURS = 205.4
TARGET_SEC = TARGET_HOURS * 3600

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
cumsum = df["duration_sec"].cumsum()
mask = cumsum <= TARGET_SEC
# include one extra row if we haven't hit target yet
if not mask.all():
    mask.iloc[mask.sum()] = True
df = df[mask].reset_index(drop=True)

print(f"After random sampling: {len(df):,}")
print(f"Total hours: {df.duration_sec.sum() / 3600:.2f}")

# =========================
# Copy filtered audio
# =========================
OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

for fname in tqdm(df["FileName"], desc="Copying audio"):
    src = AUDIO_DIR / fname
    dst = OUT_AUDIO_DIR / fname
    if not dst.exists():
        shutil.copy2(src, dst)

# =========================
# Save manifest
# =========================
df.to_csv(OUT_CSV, index=False)
print(f"Saved manifest: {OUT_CSV}")
print(f"Filtered audio directory: {OUT_AUDIO_DIR}")
