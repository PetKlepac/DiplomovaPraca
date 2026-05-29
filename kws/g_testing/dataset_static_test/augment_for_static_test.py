"""
Audio Augmentation Script for Keyword Spotting Dataset

This script processes all .wav files from testing_one_second_segments/
and generates 18 augmented versions for each file:

- 3 playback rates: 0.95, 1.0, 1.05
- 3 pitch shifts: -2, 0, +2 semitones
- 2 variants: clean and noisy

Total: 18 versions per original recording.

Processing steps:
1. Apply rate change (speed)
2. Apply pitch shift
3. Force exactly 16000 samples (1 second):
   → TRIMMING or PADDING will be clearly printed
4. Normalize speech to -22.0 dBFS
5. Clean version → saved directly
6. Noisy version:
   - Noise scaled to exact SNR = 5 dB relative to normalized speech
   - Added together (no re-normalization after mixing)

Output example:
    01_pos_finale_p-2_r0.95_clean.wav
    01_pos_finale_p0_r1.00_noisy.wav
"""

from pathlib import Path
import torch
import torchaudio
import torchaudio.functional as F
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ========================= CONFIG =========================
# Root folder
ROOT_DIR = Path(r"C:\Users\peter\Moje\Diplomka\SpracovanieNahravok\kws\h_testing_static")

# Source with your 1-second recordings
SOURCE_DIR = ROOT_DIR / "testing_one_second_segments"

# Output folder (will be created inside h_testing_static)
OUTPUT_DIR = ROOT_DIR / "testing_one_second_segments_ready"

# Noise is in the noise/ folder at root level
NOISE_FILES = [
    ROOT_DIR / "noise" / "noise_for_test_second_normalised.wav",
]

RATES = [0.95, 1.0, 1.05]
PITCHES = [-2, 0, 2]
SNR_DB = 5.0
TARGET_DBFS = -22.0
SAMPLE_RATE = 16000
TARGET_SAMPLES = 16000

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_to_dbfs(waveform: torch.Tensor, target_dbfs: float = -22.0) -> torch.Tensor:
    rms = waveform.pow(2).mean().sqrt()
    if rms == 0:
        return waveform
    current_dbfs = 20 * torch.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20.0)
    return waveform * gain

def normalize_to_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float = 5.0) -> torch.Tensor:
    speech_rms = speech.pow(2).mean().sqrt()
    if speech_rms == 0:
        return noise
    desired_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noise_rms = noise.pow(2).mean().sqrt()
    if noise_rms == 0:
        return noise
    scale = desired_noise_rms / noise_rms
    return noise * scale

def ensure_length(waveform: torch.Tensor, filename: str, target_len: int = 16000) -> torch.Tensor:
    current_len = waveform.shape[-1]
    if current_len == target_len:
        return waveform
    elif current_len > target_len:
        print(f"  → TRIMMING   {filename}  ({current_len} → {target_len} samples)")
        return waveform[..., :target_len]
    else:
        pad_len = target_len - current_len
        print(f"  → PADDING    {filename}  (+{pad_len} zero samples)")
        padding = torch.zeros((waveform.shape[0], pad_len), dtype=waveform.dtype)
        return torch.cat([waveform, padding], dim=-1)

def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype='float32')
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform.mean(dim=0, keepdim=True)  # force mono

# Pre-load noise
noises = []
for p in NOISE_FILES:
    if p.exists():
        noise_wave = load_audio(p, SAMPLE_RATE)
        noise_wave = ensure_length(noise_wave, p.name)
        noises.append(noise_wave)
    else:
        print(f"Warning: Noise file not found → {p}")

print("Starting audio augmentation...")
print("→ Speech normalized to -22.0 dBFS first")
print("→ Noise scaled to exact SNR = 5 dB")
print("→ No re-normalization after adding noise\n")

print(f"Source:  {SOURCE_DIR}")
print(f"Noise:   {ROOT_DIR / 'noise'}")
print(f"Output:  {OUTPUT_DIR}\n")

audio_files = sorted(SOURCE_DIR.glob("*.wav"))

if not audio_files:
    print("ERROR: No .wav files found in testing_one_second_segments!")
else:
    for audio_path in tqdm(audio_files, desc="Processing"):
        base_name = audio_path.stem
        waveform = load_audio(audio_path, SAMPLE_RATE)

        for rate in RATES:
            for pitch in PITCHES:
                # Rate + Pitch
                if abs(rate - 1.0) > 0.001:
                    new_sr = int(SAMPLE_RATE * rate)
                    aug_wave = F.resample(waveform, SAMPLE_RATE, new_sr)
                    aug_wave = F.resample(aug_wave, new_sr, SAMPLE_RATE)
                else:
                    aug_wave = waveform.clone()

                if pitch != 0:
                    pitch_factor = 2 ** (pitch / 12.0)
                    temp_sr = int(SAMPLE_RATE / pitch_factor)
                    temp_wave = F.resample(aug_wave, SAMPLE_RATE, temp_sr)
                    aug_wave = F.resample(temp_wave, temp_sr, SAMPLE_RATE)

                # Length enforcement + report
                aug_wave = ensure_length(aug_wave, audio_path.name)

                # Normalize speech
                clean_wave = normalize_to_dbfs(aug_wave, TARGET_DBFS)

                # Save Clean
                suffix_parts = [f"p{pitch:+.1f}".replace("+0.0", "0").replace("+", ""),
                                f"r{rate:.2f}", "clean"]
                suffix = "_" + "_".join(suffix_parts)
                sf.write(str(OUTPUT_DIR / f"{base_name}{suffix}.wav"),
                         clean_wave.squeeze().cpu().numpy(), SAMPLE_RATE, subtype='PCM_16')

                # Save Noisy
                if noises:
                    noise_idx = hash(base_name + str(rate) + str(pitch)) % len(noises)
                    noise_wave = noises[noise_idx]

                    scaled_noise = normalize_to_snr(clean_wave, noise_wave, SNR_DB)
                    noisy_wave = clean_wave + scaled_noise

                    suffix_parts[-1] = "noisy"
                    suffix = "_" + "_".join(suffix_parts)
                    sf.write(str(OUTPUT_DIR / f"{base_name}{suffix}.wav"),
                             noisy_wave.squeeze().cpu().numpy(), SAMPLE_RATE, subtype='PCM_16')

print("\nAugmentation completed!")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Total files created: {len(list(OUTPUT_DIR.glob('*.wav')))}")