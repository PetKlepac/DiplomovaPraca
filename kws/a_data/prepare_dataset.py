"""
Skript na vytvorenie datasetu pre Keyword Spotting.

Spracováva pozitívne a negatívne súbory z viacerých podpriečinkov.
Vykonáva kompletnú štandardizáciu každého audia:
  - Prevod na mono
  - Resampling na 16kHz
  - Fixnú dĺžku 16000 samples (1 sekunda) pomocou center padding / center crop
  - Normalizáciu hlasitosti na -22 dBFS

Predpokladá, že vstupné dáta už sú v štandardizovanom stave,
avšak ako prevenciu vykonáva kontrolu a automatickú úpravu.
V prípade nesprávnej dĺžky, sample rate alebo kanálov sa ihneď vypíše upozornenie.

Vytvára noisy verzie s náhodným SNR (-5 až +15 dB) iba pre tréningové dáta.
Rozdeľuje dataset na train/val podľa nastaveného pomeru.
Podporuje viaceré podpriečinky pre positive aj negative class.
"""

import os
import random
from pathlib import Path
from pydub import AudioSegment
import torch

# ────────────────────────────────────────────────────────────────
# ABSOLÚTNE CESTY K PRIEČINKOM
# ────────────────────────────────────────────────────────────────
positive_folder = r'C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\a_data\positive'
negative_folder = r'C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\a_data\negative'
noise_folder = r'C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\a_data\normalized_noises'


# ────────────────────────────────────────────────────────────────
# NASTAVENIA VÝSTUPU
# ────────────────────────────────────────────────────────────────
output_root = r'C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\b_dataset'
output_folder_name = 'd3'
train_ratio = 0.9

target_speech_dbfs = -22.0
min_snr = -5
max_snr = 15
target_sr = 16000
target_samples = 16000      # 1 sekunda pri 16kHz

random_seed = 42
random.seed(random_seed)

# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA PODPRIEČINKOV
# ────────────────────────────────────────────────────────────────
positive_subfolders_config = {
    '1sec_google_chirp_3480_final_a_finale': 1,   # počet noisy kópií na jeden originál
    '1sec_hovorene_final_a_finale': 1,
}

negative_subfolders_config = {
    '1sec_MLcommons': 1,
    # 'processed_negatives_0001-1000': 1,
    # 'processed_negatives_1001-2000': 1,
    # 'processed_negatives_2001-3000': 1,
}


def numerical_sort_key(f):
    """Vráti číselný kľúč pre správne zoradenie súborov podľa čísla v názve."""
    name = f.stem.lower() # získa názov bez prípony a prevedie na malé písmena
    try:
        num = int(''.join(filter(str.isdigit, name))) # získá iba čísla z názvu a prevedie ich na int
        return num
    except:
        return name # pokiaľ nie je číslo, vrátí celý název pre abecedné radenie


# ────────────────────────────────────────────────────────────────
# POMOCNÉ FUNKCIE PRE ŠTANDARDIZÁCIU
# ────────────────────────────────────────────────────────────────

def convert_to_mono(audio: AudioSegment, filename: str = "") -> AudioSegment:
    """1. Prevod na mono"""
    if audio.channels > 1:
        audio = audio.set_channels(1)
        print(f" ⚠ {filename:<40} mono")
    return audio


def resample_to_16kHz(audio: AudioSegment, filename: str = "") -> AudioSegment:
    """2. Resampling na 16kHz"""
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
        print(f" ⚠ {filename:<40} {target_sr}Hz")
    return audio


def normalize_loudness(audio: AudioSegment, filename: str = "") -> AudioSegment:
    """3. Normalizácia hlasitosti na -22 dBFS"""
    if abs(audio.dBFS - target_speech_dbfs) > 0.5:
        audio = audio.apply_gain(target_speech_dbfs - audio.dBFS)
        print(f" ⚠ {filename:<40} normalized")
    return audio


def fix_length_to_16000(audio: AudioSegment, filename: str = "") -> AudioSegment:
    """4. Fixná dĺžka na presne 16000 samples - opravená verzia"""
    current_samples = len(audio.get_array_of_samples())

    if current_samples == target_samples:
        return audio

    # Prevedieme na torch tensor
    samples = torch.frombuffer(
        audio.get_array_of_samples(),
        dtype=torch.int16 if audio.sample_width == 2 else torch.int32
    ).float()

    if current_samples < target_samples:
        # Center padding
        pad_total = target_samples - current_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        padded = torch.nn.functional.pad(samples, (pad_left, pad_right), mode='constant', value=0.0)
        print(f" ⚠ {filename:<40} padding (+{pad_total})")
    else:
        # Center crop
        excess = current_samples - target_samples
        cut_left = excess // 2
        padded = samples[cut_left: cut_left + target_samples]
        print(f" ⚠ {filename:<40} crop (-{excess})")

    # Vrátime späť do AudioSegment
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
noise_folder_path = Path(noise_folder)
noise_files = list(noise_folder_path.glob('*.wav'))

noise_files = sorted(noise_files, key=numerical_sort_key)

print(f"Nájdených {len(noise_files)} šumových súborov:")
for i, f in enumerate(noise_files, 1):
    print(f" {i:2d}. {f.name}")

noise_audios = []
for f in noise_files:
    try:
        audio = AudioSegment.from_file(f)
        audio = resample_to_16kHz(audio, f.stem)
        noise_audios.append((f.stem, audio))

    except Exception as e:
        print(f" ✗ Nepodarilo sa načítať {f.name}: {e}")

if not noise_audios:
    raise ValueError("Neboli načítané žiadne platné šumové súbory!")

num_noises = len(noise_audios)
print(f"\nCelkovo načítaných šumov: {num_noises}\n")


# ────────────────────────────────────────────────────────────────
# NASTAVENIE VÝSTUPNEJ CESTY
# ────────────────────────────────────────────────────────────────
output_base_path = Path(output_root) / output_folder_name
os.makedirs(output_base_path, exist_ok=True)

print(f"Výstup bude uložený do: {output_base_path.absolute()}\n")


def get_output_dir(split: str, class_name: str, subfolder: str) -> Path:
    """Vytvorí a vráti cieľový priečinok pre daný split, triedu a podpriečinok."""
    dir_path = output_base_path / split / class_name / subfolder
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


# ────────────────────────────────────────────────────────────────
# ULOŽENIE INFORMÁCIÍ O DATASETE
# ────────────────────────────────────────────────────────────────
def save_dataset_info(output_path: Path):
    """Uloží informácie o konfigurácii datasetu do textového súboru."""
    info_file = output_path / "dataset_info.txt"

    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=== INFORMÁCIE O DATASETE ===\n\n")
        f.write(f"Názov datasetu:          {output_folder_name}\n")
        f.write(f"Pomer train/val:         {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%\n")
        f.write(f"Random seed:             {random_seed}\n")
        f.write(f"Cieľová hlasitosť reči:  {target_speech_dbfs} dBFS\n")
        f.write(f"Rozsah SNR:              {min_snr} až {max_snr} dB\n\n")

        f.write("=== POSITIVE SUBFOLDERS ===\n")
        for folder, copies in positive_subfolders_config.items():
            f.write(f"  • {folder}  →  {copies} noisy kópií\n")

        f.write("\n=== NEGATIVE SUBFOLDERS ===\n")
        for folder, copies in negative_subfolders_config.items():
            f.write(f"  • {folder}  →  {copies} noisy kópií\n")

        f.write("\n=== ĎALŠIE INFORMÁCIE ===\n")
        f.write(f"Počet načítaných šumov:  {num_noises}\n")
        f.write(f"Dátum vytvorenia:        {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Informácie o datasete uložené do: {info_file.name}")


# ────────────────────────────────────────────────────────────────
# HLAVNÉ SPRACOVANIE
# ────────────────────────────────────────────────────────────────
save_dataset_info(output_base_path) # Uložíme informácie hneď po vytvorení výstupného priečinka

noise_idx = 0 # counter, ktorý si pamätá, od ktorého šumu momentálne začíname pri vytváraní noisy verzií

for class_name, subfolders_config in [('positive', positive_subfolders_config), ('negative', negative_subfolders_config)]:
    input_base = Path(positive_folder if class_name == 'positive' else negative_folder)

    print(f"\n{'=' * 85}")
    print(f"SPRACOVÁVAM TRIEDU: {class_name.upper()}")
    print(f"{'=' * 85}\n")

    for subfolder_name, noisy_copies_train in subfolders_config.items():
        subfolder_path = input_base / subfolder_name

        if not subfolder_path.exists():
            print(f" ⚠ Preskakujem {class_name}/{subfolder_name} → priečinok neexistuje")
            continue

        all_files = list(subfolder_path.glob('*.wav'))
        if not all_files:
            print(f" ⚠ V priečinku {class_name}/{subfolder_name} nie sú žiadne .wav súbory")
            continue

        print(f"Nájdených {len(all_files)} súborov v {class_name}/{subfolder_name}")

        # náhodné rozdelenie na train a val
        random.shuffle(all_files)
        split_idx = int(len(all_files) * train_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f" → Train súbory: {len(train_files)} | Val súbory: {len(val_files)}")

        # ────────────────────────────────────────────────────────────────
        # VALIDATION SPLIT
        # ────────────────────────────────────────────────────────────────
        if val_files:
            val_dir = get_output_dir('val', class_name, subfolder_name)
            print(f" → Kopírujem a štandardizujem {len(val_files)} súborov do validácie...")

            for filename in val_files:
                base_name = filename.stem
                try:
                    voice_audio = AudioSegment.from_file(filename)
                    voice_audio = convert_to_mono(voice_audio, base_name)
                    voice_audio = resample_to_16kHz(voice_audio, base_name)
                    voice_audio = normalize_loudness(voice_audio, base_name)
                    voice_audio = fix_length_to_16000(voice_audio, base_name)

                    final_samples = len(voice_audio.get_array_of_samples())
                    if final_samples != target_samples:
                        print(f" ✗ KRITICKÁ CHYBA: {final_samples} samples u {base_name}")

                    # Uložíme ako original
                    output_path = val_dir / f"{base_name}_original.wav"
                    voice_audio.export(output_path, format='wav')

                except Exception as e:
                    print(f" ✗ Chyba pri validácii {filename.name}: {e}")
                    continue

        # ────────────────────────────────────────────────────────────────
        # TRAINING SPLIT
        # ────────────────────────────────────────────────────────────────
        if train_files:
            train_dir = get_output_dir('train', class_name, subfolder_name)
            print(f" → Spracovávam tréningovú časť → normalizácia + originál + {noisy_copies_train} noisy verzií")

            for filename in train_files:
                base_name = filename.stem
                try:
                    # načítanie + kompletná štandardizácia
                    voice_audio = AudioSegment.from_file(filename)
                    voice_audio = convert_to_mono(voice_audio, base_name)
                    voice_audio = resample_to_16kHz(voice_audio, base_name)
                    voice_audio = normalize_loudness(voice_audio, base_name)
                    voice_audio = fix_length_to_16000(voice_audio, base_name)

                    # uloženie normalizovaného originálu
                    normalized_filename = f"{base_name}_norm.wav"
                    voice_audio.export(train_dir / normalized_filename, format='wav')

                    # === VYTVORENIE NOISY KÓPIÍ ===
                    for copy_num in range(1, noisy_copies_train + 1):
                        list_index = noise_idx % num_noises
                        noise_name, noise_audio = noise_audios[list_index]

                        final_duration = len(voice_audio)  # už je štandardizované (16000 samples)

                        # predĺženie šumu podľa finálnej dĺžky
                        if len(noise_audio) < final_duration:
                            noise_audio = noise_audio * ((final_duration // len(noise_audio)) + 2)

                        start_time = random.randint(0, len(noise_audio) - final_duration)
                        noise_segment = noise_audio[start_time: start_time + final_duration]

                        # aplikácia SNR
                        snr = random.uniform(min_snr, max_snr)
                        target_noise_db = voice_audio.dBFS - snr
                        noise_segment = noise_segment.apply_gain(target_noise_db)

                        mixed_audio = voice_audio.overlay(noise_segment)

                        final_samples = len(mixed_audio.get_array_of_samples())
                        if final_samples != target_samples:
                            print(f" ✗ KRITICKÁ CHYBA: {final_samples} samples u {base_name}")

                        output_filename = f"{base_name}_noisy_{copy_num}.wav"
                        mixed_audio.export(train_dir / output_filename, format='wav')

                        noise_idx += 1

                except Exception as e:
                    print(f" ✗ Chyba pri spracovaní {filename.name}: {e}")
                    continue

print("\n=== VYTÁRANIE DATASETU ÚSPEŠNE DOKONČENÉ ===")
print(f"Dataset bol uložený do: {output_base_path}")


