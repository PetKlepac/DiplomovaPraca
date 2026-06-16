""" Skript na generovanie sliding window grafov pre každý sample """

from pathlib import Path
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from kws.c_model.m_w2v import Wav2VecKWS

# ────────────────────────────────────────────────────────────────
# NASTAVITELNÉ PARAMETRE
# ────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.05
SAMPLE_RATE = 16000
THRESHOLD = 0.5
TARGET_DBFS = -22.0

# Nastavenia grafov
CHART_FIGSIZE = (14, 6)
CHART_DPI = 180

# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "d9_m_w2v_c"  # ← ZMEŇ podľa potreby

project_root = Path(__file__).resolve().parents[3]
root_folder = project_root / "kws" / "g_testing" / "dynamic" / "data"
model_path = project_root / "kws" / "h_result" / MODEL_NAME / f"{MODEL_NAME}.pth"

output_dir = project_root / "kws" / "h_result" / MODEL_NAME / "testing"
charts_dir = output_dir / "per_sample_sliding_charts"
charts_dir.mkdir(parents=True, exist_ok=True)

print(f"Output charts folder: {charts_dir}")
print(f"Model: {model_path.name}\n")

# ────────────────────────────────────────────────────────────────
# NAČÍTANIE MODELU
# ────────────────────────────────────────────────────────────────
print("Načítavam model...")
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
def normalize_loudness(waveform: torch.Tensor, target_dbfs: float = TARGET_DBFS) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms.item() <= 1e-8:
        return torch.zeros_like(waveform)
    current_dbfs = 20 * torch.log10(rms)
    gain_db = target_dbfs - current_dbfs.item()
    gain_linear = 10 ** (gain_db / 20.0)
    return waveform * gain_linear


def sliding_window_inference(waveform: torch.Tensor, window_samples: int, step_samples: int):
    probs = []
    length = len(waveform)
    for start in range(0, length - window_samples + 1, step_samples):
        window = waveform[start:start + window_samples]
        if len(window) < window_samples:
            pad = torch.zeros(window_samples - len(window), dtype=waveform.dtype)
            window = torch.cat([window, pad])
        if window.ndim == 1:
            window = window.unsqueeze(0)

        with torch.no_grad():
            logits = model(window.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append((start / SAMPLE_RATE, prob))
    return probs


def process_and_save_chart(audio_path: Path, global_id: int):
    try:
        waveform_np, orig_sr = sf.read(str(audio_path), dtype="float32")
        waveform = torch.from_numpy(waveform_np).float()

        if orig_sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        waveform = normalize_loudness(waveform)

        window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        step_samples = int(STEP_SECONDS * SAMPLE_RATE)

        window_probs = sliding_window_inference(waveform, window_samples, step_samples)

        if not window_probs:
            return

        times, probs = zip(*window_probs)
        folder_name = audio_path.parent.name

        plt.figure(figsize=CHART_FIGSIZE)
        plt.plot(times, probs, color='blue', linewidth=1.8, alpha=0.85, label='Probability (keyword)')
        plt.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold = {THRESHOLD}')
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)

        max_p = max(probs)
        max_t = times[probs.index(max_p)]
        plt.scatter([max_t], [max_p], color='orange', s=60, zorder=5, label=f'Max: {max_p:.4f}')

        plt.title(f'Sliding Window Probabilities — {folder_name.upper()} / {audio_path.name}\n'
                  f'ID: {global_id:03d} | Duration: {len(waveform) / SAMPLE_RATE:.2f}s',
                  fontsize=14, pad=20)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Keyword Probability', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Highlight vysokých hodnôt
        for t, p in zip(times, probs):
            if p > 0.7:
                plt.text(t, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

        safe_name = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in audio_path.name)
        chart_file = charts_dir / f"{global_id:03d}_{safe_name}_sliding.png"

        plt.savefig(chart_file, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Chyba pri {audio_path.name}: {e}")


# ────────────────────────────────────────────────────────────────
# HLAVNÉ SPRACOVANIE
# ────────────────────────────────────────────────────────────────
folders = ['positive', 'negative']
global_idx = 1

print("Začínam generovať sliding window grafy...\n")

for folder_name in folders:
    folder_path = root_folder / folder_name
    if not folder_path.exists():
        print(f"Priečinok {folder_name} neexistuje → preskakujem")
        continue

    audio_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == '.wav'])
    print(f"Spracovávam {folder_name} — {len(audio_files)} súborov")

    for audio_path in tqdm(audio_files, desc=folder_name):
        process_and_save_chart(audio_path, global_idx)
        global_idx += 1

print(f"\n✅ Všetky sliding window grafy boli uložené do:")
print(f"{charts_dir}")
print("Hotovo!")