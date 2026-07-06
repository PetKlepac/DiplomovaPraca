"""
Skript na testovanie Whisper baseline na dlhších audio segmentoch.

- Načíta model faster-whisper
- Pre každý audio súbor vykoná transkripciu
- Uloží plný text + fuzzy score ako max_prob (normalizovaný do 0-1)
"""

from pathlib import Path
import torch
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
KEYWORD_VARIATIONS = ["finále"]
SAMPLE_RATE = 16000
TARGET_DBFS = -22.0

MODEL_NAME = "whisper"
WHISPER_MODEL_SIZE = "large-v3"

# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
root_folder = project_root / "kws" / "b_dataset" / "testing"

output_base = project_root / "kws" / "g_result" / MODEL_NAME / "testing"
output_dir = output_base / MODEL_NAME
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Whisper model: {WHISPER_MODEL_SIZE}")
print(f"Výstupný priečinok: {output_dir}\n")

# ────────────────────────────────────────────────────────────────
# NAČÍTANIE MODELU
# ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Načítavam model {WHISPER_MODEL_SIZE} ...")
model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
print("Model načítaný.\n")


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


def process_file(audio_path: Path):
    try:
        waveform_np, orig_sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(waveform_np).float()
        waveform = convert_to_mono(waveform)
        waveform = normalize_loudness(waveform)

        segments, info = model.transcribe(
            waveform.numpy(),
            language="cs",
            initial_prompt="Letecká komunikácia. Kľúčové slovo je finále.",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=0.6,
        )

        full_text = ""
        max_fuzzy = 0.0

        for segment in segments:
            full_text += segment.text + " "
            for word_info in segment.words:
                word_clean = word_info.word.lower().strip(".,!? ")

                # Fuzzy matching
                best_similarity = max(
                    fuzz.ratio(word_clean, variant) for variant in KEYWORD_VARIATIONS
                )

                if best_similarity > max_fuzzy:
                    max_fuzzy = best_similarity

        # Normalizácia na rozsah 0-1
        max_prob = max_fuzzy / 100.0

        return {
            'filename': audio_path.name,
            'max_prob': round(max_prob, 4),
            'whisper_text': full_text.strip(),
            'duration_sec': round(len(waveform) / SAMPLE_RATE, 2),
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
# VÝSTUP
# ────────────────────────────────────────────────────────────────
if all_results:
    df = pd.DataFrame(all_results)
    cols = ['global_id', 'folder', 'filename', 'max_prob', 'whisper_text', 'duration_sec']
    csv_path = output_dir / f"table_{MODEL_NAME}.csv"
    df[cols].to_csv(csv_path, index=False, encoding='utf-8')

    positive_count = len(df[df['folder'] == 'positive'])
    negative_count = len(df[df['folder'] == 'negative'])

    print("\n" + "="*70)
    print("SÚHRN SPRACOVANIA (Whisper baseline)")
    print("="*70)
    print(f" Positive segmenty : {positive_count:>3}")
    print(f" Negative segmenty : {negative_count:>3}")
    print(f" Spolu : {len(df):>3}")
    print("="*70)
    print(f"\nTabuľka uložená: {csv_path}")
    print(f"Všetky výsledky sú v: {output_dir}")
else:
    print("Žiadne súbory spracované.")