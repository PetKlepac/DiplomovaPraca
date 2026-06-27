"""
Skript na testovanie modelu Wav2VecKWS na dlhých audio segmentoch pomocou sliding window.

Tento skript vykonáva inferenciu na dlhších audio súboroch tak, že:
- Načíta natrénovaný model Wav2VecKWS
- Aplikuje sliding window inferenciu (1-sekundové okná s malým krokom)
- Pre každý audio súbor nájde najvyššiu pravdepodobnosť kľúčového slova
- Uloží objektívne výsledky do CSV tabuľky v samostatnom priečinku podľa epochy

Vstup:
    Priečinky s testovacími audio súbormi (positive/ a negative/)

Výstup:
    CSV súbor s výsledkami (table_*.csv) v priečinku podľa epochy
"""

from pathlib import Path
import torch
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
from kws.c_model.m_w2v import Wav2VecKWS
import torchaudio
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.05
SAMPLE_RATE = 16000
TARGET_DBFS = -22.0

MODEL_NAME = "d4_m_w2v_c"
EPOCH_TO_USE = 10


# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
root_folder = project_root / "kws" / "a_data_preparation" / "prepared_data" / "testing"
model_dir = project_root / "kws" / "g_result" / MODEL_NAME / "models"
model_path = model_dir / f"{MODEL_NAME}_e{EPOCH_TO_USE:02d}.pth"

output_base = project_root / "kws" / "g_result" / MODEL_NAME / "testing"
file_suffix = f"{MODEL_NAME}_e{EPOCH_TO_USE:02d}"

# Vytvoríme samostatný priečinok pre tento beh
output_dir = output_base / file_suffix
output_dir.mkdir(parents=True, exist_ok=True)

# Kontrola modelu
if not model_path.exists():
    print(f"Model nenájdený: {model_path}")
    print("Dostupné modely:")
    for p in sorted(model_dir.glob("*.pth")):
        print(f"  {p.name}")
    raise FileNotFoundError(f"Model {model_path.name} neexistuje!")

print(f"Model: {model_path.name}")
print(f"Výstupný priečinok: {output_dir}\n")


# ────────────────────────────────────────────────────────────────
# NAČÍTANIE MODELU
# ────────────────────────────────────────────────────────────────
print("Načítavam Wav2Vec2 model...")
model = Wav2VecKWS(num_classes=2)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model načítaný.\n")


# ────────────────────────────────────────────────────────────────
# FUNKCIE
# ────────────────────────────────────────────────────────────────
def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    return waveform


def normalize_loudness(waveform: torch.Tensor, target_dbfs: float = TARGET_DBFS) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    current_dbfs = 20 * torch.log10(rms)
    gain_db = target_dbfs - current_dbfs.item()
    gain_linear = 10 ** (gain_db / 20.0)
    return waveform * gain_linear


def resample_to_16kHz(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    waveform = resampler(waveform)
    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    return waveform


def sliding_window_inference(waveform: torch.Tensor, window_samples: int, step_samples: int):
    probs = []
    length = len(waveform) if waveform.ndim == 1 else waveform.shape[1]

    for start in range(0, length, step_samples):
        end = min(start + window_samples, length)
        if waveform.ndim == 1:
            window = waveform[start:end]
        else:
            window = waveform[:, start:end]

        current_len = len(window) if waveform.ndim == 1 else window.shape[1]
        if current_len < window_samples:
            pad_length = window_samples - current_len
            if waveform.ndim == 1:
                pad = torch.zeros(pad_length, dtype=waveform.dtype)
                window = torch.cat([window, pad])
            else:
                pad = torch.zeros((window.shape[0], pad_length), dtype=waveform.dtype)
                window = torch.cat([window, pad], dim=1)

        if window.ndim == 1:
            window = window.unsqueeze(0)

        with torch.no_grad():
            logits = model(window.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append((start / SAMPLE_RATE, prob))

    return probs


def process_file(audio_path: Path):
    try:
        waveform_np, orig_sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(waveform_np).float()

        waveform = convert_to_mono(waveform)
        waveform = resample_to_16kHz(waveform, orig_sr, SAMPLE_RATE)
        waveform = normalize_loudness(waveform)

        window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        step_samples = int(STEP_SECONDS * SAMPLE_RATE)

        window_probs = sliding_window_inference(waveform, window_samples, step_samples)

        if not window_probs:
            return None

        probs_only = [p[1] for p in window_probs]
        max_prob = max(probs_only)
        max_idx = np.argmax(probs_only)
        max_time_sec = window_probs[max_idx][0]

        return {
            'filename': audio_path.name,
            'max_prob': round(max_prob, 4),
            'max_time_sec': round(max_time_sec, 2),
            'duration_sec': round(len(waveform) / SAMPLE_RATE if waveform.ndim == 1 else waveform.shape[1] / SAMPLE_RATE, 2),
        }

    except Exception as e:
        print(f"Chyba pri {audio_path.name}: {e}")
        return None


# ────────────────────────────────────────────────────────────────
# HLAVNÉ SPRACOVANIE
# ────────────────────────────────────────────────────────────────
folders = ['positive', 'negative']
all_results = []
global_idx = 1

for folder_name in folders:
    folder_path = root_folder / folder_name
    if not folder_path.exists():
        print(f"Priečinok {folder_name} neexistuje → preskakujem")
        continue

    audio_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == '.wav'])
    print(f"Spracovávam {folder_name} — {len(audio_files)} súborov")

    for audio_path in tqdm(audio_files, desc=folder_name):
        result = process_file(audio_path)
        if result:
            result['folder'] = folder_name
            result['global_id'] = f"{global_idx:03d}"
            all_results.append(result)
            global_idx += 1


# ────────────────────────────────────────────────────────────────
# VÝSTUP – IBA CSV (do samostatného priečinka podľa epochy)
# ────────────────────────────────────────────────────────────────
if all_results:
    df = pd.DataFrame(all_results)
    cols = ['global_id', 'folder', 'filename', 'max_prob', 'max_time_sec', 'duration_sec']

    csv_path = output_dir / f"table_{file_suffix}.csv"
    df[cols].to_csv(csv_path, index=False, encoding='utf-8')

    # ── Súhrn podľa tried ───────────────────────────────────────
    positive_count = len(df[df['folder'] == 'positive'])
    negative_count = len(df[df['folder'] == 'negative'])

    print("\n" + "="*65)
    print("SÚHRN SPRACOVANIA")
    print("="*65)
    print(f"  Positive segmenty : {positive_count:>3}")
    print(f"  Negative segmenty : {negative_count:>3}")
    print(f"  Spolu             : {len(df):>3}")
    print("="*65)

    print(f"\nTabuľka uložená: {csv_path}")
    print(f"Všetky výsledky sú v: {output_dir}")
else:
    print("Žiadne súbory spracované.")