"""
Kombinovaný skript na vytvorenie finálneho tréningového datasetu pre Keyword Spotting.

Účel:
- Spracuje pozitívne a negatívne vzorky z rôznych podpriečinkov
- Vytvorí train/val split.
- Pre validáciu ukladá čisté verzie
- Pre tréning vytvára clean verzie + viaceré noisy verzie (s náhodným SNR)
- Všetky nahrávky štandardizuje (mono, 16kHz, -22 dBFS, presne 1 sekunda)

Výstupná štruktúra:
d1/
├── train/
│   ├── positive/
│   └── negative/
└── val/
    ├── positive/
    └── negative/
"""

import random
import numpy as np
import torch
from pathlib import Path
from pydub import AudioSegment

# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIVE_FOLDER = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "positive"
NEGATIVE_FOLDER = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "negative"
NOISE_FOLDER = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "noises"

OUTPUT_FOLDER = PROJECT_ROOT / "kws" / "b_dataset"
output_folder_name = 'd2'

train_val_split = 0.9
TARGET_RMS_DBFS = -22.0
min_snr = 5
max_snr = 15
target_sr = 16000
target_samples = 16000

random_seed = 42
random.seed(random_seed)

# ────────────────────────────────────────────────────────────────
# VÝBER PODPRIEČINKOV
# ────────────────────────────────────────────────────────────────
positive_subfolders_config = {
    'airport_positive_140': {'noisy_copies': 11, 'create_clean': True},
}

negative_subfolders_config = {
    'airport_negative_nedele_500': {'noisy_copies': 11, 'create_clean': True},
    'airport_negative_sobotanedele_500': {'noisy_copies': 11, 'create_clean': True},
    'airport_negative_pondeli_500': {'noisy_copies': 11, 'create_clean': True},
    'airport_negative_utery_500': {'noisy_copies': 11, 'create_clean': True},
    'airport_negative_streda_500': {'noisy_copies': 11, 'create_clean': True},
}


# ────────────────────────────────────────────────────────────────
# POMOCNÉ FUNKCIE
# ────────────────────────────────────────────────────────────────
def convert_to_mono(audio: AudioSegment, filename: str = "") -> AudioSegment:
    if audio.channels > 1:
        audio = audio.set_channels(1)
        if filename:
            print(f" ⚠ {filename:<40} → mono")
    return audio


def resample_to_16kHz(audio: AudioSegment, filename: str = "") -> AudioSegment:
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
        if filename:
            print(f" ⚠ {filename:<40} → 16kHz")
    return audio


def normalize_to_rms(audio: AudioSegment, target_rms_dbfs: float = -23.0, filename: str = "") -> AudioSegment:
    if len(audio) == 0:
        return audio

    samples = np.array(audio.get_array_of_samples()).astype(np.float64)
    rms = np.sqrt(np.mean(samples ** 2) + 1e-10)

    if rms < 1e-6:
        if filename:
            print(f" ⚠ {filename:<40} príliš tiché, RMS norm preskočená")
        return audio

    current_rms_dbfs = 20 * np.log10(rms / 32768.0)
    gain_db = target_rms_dbfs - current_rms_dbfs
    normalized = audio.apply_gain(gain_db)

    if filename and abs(gain_db) > 0.1:
        print(f" ⚠ {filename:<40} normalized")

    return normalized


def fix_length_to_16000(audio: AudioSegment, filename: str = "") -> AudioSegment:
    current_samples = len(audio.get_array_of_samples())
    if current_samples == target_samples:
        return audio

    samples = torch.frombuffer(
        audio.get_array_of_samples(),
        dtype=torch.int16 if audio.sample_width == 2 else torch.int32
    ).float()

    if current_samples < target_samples:
        pad_total = target_samples - current_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = torch.nn.functional.pad(samples, (pad_left, pad_right), mode='constant', value=0.0)
        print(f" ⚠ {filename:<40} padding (+{pad_total})")
    else:
        excess = current_samples - target_samples
        cut_left = excess // 2
        padded = samples[cut_left: cut_left + target_samples]
        print(f" ⚠ {filename:<40} crop (-{excess})")

    padded_int = padded.short() if audio.sample_width == 2 else padded.int()
    new_audio = AudioSegment(
        padded_int.numpy().tobytes(),
        frame_rate=target_sr,
        sample_width=audio.sample_width,
        channels=1
    )
    return new_audio


# ────────────────────────────────────────────────────────────────
# NAČÍTANIE ŠUMOV
# ────────────────────────────────────────────────────────────────
noise_folder_path = Path(NOISE_FOLDER)
noise_files = sorted(noise_folder_path.glob('*.wav'), key=lambda f: int(''.join(filter(str.isdigit, f.stem))) if any(
    c.isdigit() for c in f.stem) else f.stem)

print(f"Nájdených {len(noise_files)} šumových súborov.")
noise_audios = []
for f in noise_files:
    try:
        audio = AudioSegment.from_file(f)
        audio = convert_to_mono(audio, f.stem)
        audio = resample_to_16kHz(audio, f.stem)
        audio = normalize_to_rms(audio, TARGET_RMS_DBFS, f.stem)
        noise_audios.append((f.stem, audio))
    except Exception as e:
        print(f" ✗ Nepodarilo sa načítať {f.name}: {e}")

num_noises = len(noise_audios)
print(f"Celkovo načítaných šumov: {num_noises}\n")

# ────────────────────────────────────────────────────────────────
# NASTAVENIE VÝSTUPNEJ CESTY
# ────────────────────────────────────────────────────────────────
output_base_path = Path(OUTPUT_FOLDER) / output_folder_name
output_base_path.mkdir(parents=True, exist_ok=True)
print(f"Výstup bude uložený do: {output_base_path.absolute()}\n")


def get_output_dir(split: str, class_name: str, subfolder: str) -> Path:
    dir_path = output_base_path / split / class_name / subfolder
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

# ────────────────────────────────────────────────────────────────
# ULOŽENIE INFORMÁCIÍ O DATASETE
# ────────────────────────────────────────────────────────────────
def save_dataset_info(output_path: Path):
    info_file = output_path / "dataset_info.txt"
    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=== INFORMÁCIE O DATASETE ===\n\n")
        f.write(f"Názov datasetu: {output_folder_name}\n")
        f.write(f"Pomer train/val: {train_val_split * 100:.0f}% / {(1 - train_val_split) * 100:.0f}%\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Cieľová hlasitosť: {TARGET_RMS_DBFS} dBFS\n")
        f.write(f"Rozsah SNR: {min_snr} až {max_snr} dB\n\n")

        f.write("=== POSITIVE SUBFOLDERS ===\n")
        for folder, cfg in positive_subfolders_config.items():
            f.write(f" • {folder} → {cfg['noisy_copies']} noisy | clean: {cfg['create_clean']}\n")

        f.write("\n=== NEGATIVE SUBFOLDERS ===\n")
        for folder, cfg in negative_subfolders_config.items():
            f.write(f" • {folder} → {cfg['noisy_copies']} noisy | clean: {cfg['create_clean']}\n")

        f.write(f"\nPočet šumov: {num_noises}\n")
        f.write(f"Dátum: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Informácie o datasete uložené do: {info_file.name}")

# ────────────────────────────────────────────────────────────────
# HLAVNÉ SPRACOVANIE
# ────────────────────────────────────────────────────────────────
save_dataset_info(output_base_path)

noise_idx = 0

for class_name, subfolders_config in [('positive', positive_subfolders_config),
                                      ('negative', negative_subfolders_config)]:
    input_base = POSITIVE_FOLDER if class_name == 'positive' else NEGATIVE_FOLDER
    print(f"\n{'=' * 90}")
    print(f"SPRACOVÁVAM TRIEDU: {class_name.upper()}")
    print(f"{'=' * 90}\n")

    for subfolder_name, config in subfolders_config.items():
        noisy_copies_train = config['noisy_copies']
        create_clean = config['create_clean']
        subfolder_path = input_base / subfolder_name

        if not subfolder_path.exists():
            print(f" ⚠ Preskakujem {class_name}/{subfolder_name}")
            continue

        all_files = list(subfolder_path.glob('*.wav'))
        if not all_files:
            print(f" ⚠ V priečinku {class_name}/{subfolder_name} nie sú žiadne súbory")
            continue

        print(f"Nájdených {len(all_files)} súborov v {class_name}/{subfolder_name}")
        random.shuffle(all_files)

        split_idx = int(len(all_files) * train_val_split)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f" → Train: {len(train_files)} | Val: {len(val_files)}")

        # VALIDATION - clean verzie
        if val_files:
            val_dir = get_output_dir('val', class_name, subfolder_name)
            print(f" → Spracovávam {len(val_files)} súborov do validácie...")
            for filename in val_files:
                base_name = filename.stem
                try:
                    voice_audio = AudioSegment.from_file(filename)
                    voice_audio = convert_to_mono(voice_audio, base_name)
                    voice_audio = resample_to_16kHz(voice_audio, base_name)
                    voice_audio = normalize_to_rms(voice_audio, TARGET_RMS_DBFS, base_name)
                    voice_audio = fix_length_to_16000(voice_audio, base_name)

                    output_path = val_dir / f"{base_name}_original.wav"
                    voice_audio.export(output_path, format='wav')
                except Exception as e:
                    print(f" ✗ Chyba pri validácii {filename.name}: {e}")

        # TRAINING - clean + noisy
        if train_files:
            train_dir = get_output_dir('train', class_name, subfolder_name)
            print(f" → Train: normalizácia + {'clean + ' if create_clean else ''}{noisy_copies_train} noisy verzií")

            for filename in train_files:
                base_name = filename.stem
                try:
                    voice_audio = AudioSegment.from_file(filename)
                    voice_audio = convert_to_mono(voice_audio, base_name)
                    voice_audio = resample_to_16kHz(voice_audio, base_name)
                    voice_audio = normalize_to_rms(voice_audio, TARGET_RMS_DBFS, base_name)
                    voice_audio = fix_length_to_16000(voice_audio, base_name)

                    if create_clean:
                        voice_audio.export(train_dir / f"{base_name}_norm.wav", format='wav')

                    # Noisy verzie
                    for copy_num in range(1, noisy_copies_train + 1):
                        list_index = noise_idx % num_noises
                        _, noise_audio = noise_audios[list_index]

                        final_duration = len(voice_audio)
                        if len(noise_audio) < final_duration:
                            noise_audio = noise_audio * ((final_duration // len(noise_audio)) + 2)

                        start_time = random.randint(0, len(noise_audio) - final_duration)
                        noise_segment = noise_audio[start_time:start_time + final_duration]

                        snr = random.uniform(min_snr, max_snr)
                        gain_db = -snr
                        noise_segment = noise_segment.apply_gain(gain_db)

                        mixed_audio = voice_audio.overlay(noise_segment)
                        mixed_audio.export(train_dir / f"{base_name}_noisy_{copy_num}.wav", format='wav')

                        noise_idx += 1

                except Exception as e:
                    print(f" ✗ Chyba pri spracovaní {filename.name}: {e}")



print("\n=== VYTÁRANIE DATASETU ÚSPEŠNE DOKONČENÉ ===")
print(f"Dataset bol uložený do: {output_base_path}")