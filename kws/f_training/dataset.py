"""Dataset pre načítavanie surových waveformov (.wav) pre Keyword Spotting."""

import random
import torch
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path


class RawWaveformDataset(Dataset):
    """Dataset pre načítavanie surových .wav súborov určený pre Wav2Vec2.
    Používa knižnicu soundfile (spoľahlivá na Windows) a voliteľne torchaudio na resampling.
    Pridáva mierne augmentácie."""

    def __init__(
        self,
        subdir: str,                    # "train" or "val"
        root_dir: str | Path,           # koreňový priečinok s datasetom
        target_samples: int = 16000,    # cieľová dĺžka waveformu v vzorkách
        augment_training: bool = True,  # zapne augmentácie iba pri tréningu
    ):
        self.root_dir = Path(root_dir)                      # prevod na Path objekt
        self.base_dir = self.root_dir / subdir              # priečinok train alebo val
        self.target_samples = target_samples                # uloženie cieľovej dĺžky
        self.augment_training = augment_training            # nastavenie augmentácií
        self.samples = self._build_samples()                # zoznam všetkých súborov
        print(f"✅ Loaded {len(self.samples)} samples from {self.base_dir} "
              f"(augment={augment_training})")

    def _build_samples(self):
        """Zozbiera všetky .wav súbory z priečinkov positive/ a negative/"""
        samples = []
        for label_name, label in [("positive", 1), ("negative", 0)]:
            label_dir = self.base_dir / label_name
            if not label_dir.exists():
                print(f"Warning: Directory not found → {label_dir}")
                continue

            # Prechádzame všetky súbory (aj v podpriečinkoch)
            for wav_file in label_dir.rglob("*.wav"):       # rglob = rekurzívne
                samples.append((wav_file, label))

        if len(samples) == 0:
            raise ValueError(f"No .wav files found in {self.base_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Načíta jednu nahrávku, spracuje ju a vráti waveform + label."""
        wav_path, label = self.samples[idx]
        try:
            # načítanie súboru
            waveform_np, orig_sr = sf.read(str(wav_path), dtype="float32")
            waveform = torch.from_numpy(waveform_np).float()

            # augmentácie (len pre trénovacie dáta)
            if self.augment_training:
                # 1. Random Gain
                if random.random() < 0.6:
                    gain = random.uniform(0.8, 1.2)
                    waveform = waveform * gain                  # náhodné zosilnenie/ztlmenie

                # 2. Time Roll (posun bez paddingu)
                if random.random() < 0.4:
                    shift = random.randint(-400, 400)           # ±25ms pri 16kHz
                    waveform = torch.roll(waveform, shifts=shift)

                # 3. Light Background Noise
                if random.random() < 0.3:
                    noise_amp = random.uniform(0.003, 0.018)
                    noise = torch.randn_like(waveform) * noise_amp
                    waveform = waveform + noise

            if len(waveform) != self.target_samples:
                print(f" ⚠️ Dĺžka {len(waveform)} samples | {wav_path.name}, "
                      f"očakávané: {self.target_samples} samples")

            return waveform, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"⚠️ Chyba pri načítaní {wav_path.name}: {e}")
            return None, None