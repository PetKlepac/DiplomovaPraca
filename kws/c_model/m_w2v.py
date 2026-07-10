"""
Wav2VecKWS – implementácia Wav2Vec2 pre Keyword Spotting (KWS).

Trieda poskytuje jednoduché rozhranie na:
- Načítanie / stiahnutie predtrénovaného Wav2Vec2 modelu
- Zmrazovanie feature extraktora a enkodér vrstiev
- Odmrazenie posledných N vrstiev encoderu
- Klasifikáciu (detekciu kľúčového slova) na 1-sekundových waveforms
- Lokálne ukladanie modelu (pre opakované použitie bez sťahovania)
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor


class Wav2VecKWS(nn.Module):
    """
    Wav2Vec2 model prispôsobený na detekciu kľúčových slov.
    """
    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "fav-kky/wav2vec2-base-cs-80k-ClTRUS",
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 3,
        verbose: bool = True,
    ):
        super().__init__()

        self.local_model_path = Path(__file__).parent / "pretrained_wav2vec"
        self.verbose = verbose
        self.freeze_feature_extractor = freeze_feature_extractor
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers

        # Načítanie alebo stiahnutie modelu
        if not self.local_model_path.exists() or not (self.local_model_path / "config.json").exists():
            if self.verbose:
                print(f"Model '{model_name}' nenájdený lokálne → sťahujem...")
            self._download_and_save_model(model_name)
        else:
            if self.verbose:
                print(f"Načítavam model z: {self.local_model_path}")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.local_model_path)
        self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.local_model_path,
            num_labels=num_classes,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        ).to("cuda")

        self._apply_freezing_strategy()

        if self.verbose:
            print(f"Feature extractor frozen: {self.freeze_feature_extractor}")
            print(f"Encoder frozen: {self.freeze_encoder} "
                  f"(unfrozen last {self.unfreeze_last_n_layers} layers)")

    def _download_and_save_model(self, model_name: str):
        """Stiahne model z Hugging Face a uloží ho lokálne pre budúce použitie."""
        self.local_model_path.mkdir(parents=True, exist_ok=True)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        feature_extractor.save_pretrained(self.local_model_path)
        model.save_pretrained(self.local_model_path)

        if self.verbose:
            print(f"Model úspešne stiahnutý a uložený do: {self.local_model_path}")

    def _apply_freezing_strategy(self):
        """Aplikuje stratégiu freezovania podľa nastavení v __init__."""
        # Zmrazovanie feature extractora
        if self.freeze_feature_extractor:
            for param in self.wav2vec.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        encoder = self.wav2vec.wav2vec2.encoder

        # Zmrazovanie celého encoderu + odmrazenie posledných N vrstiev
        if self.freeze_encoder and self.unfreeze_last_n_layers > 0:
            for param in encoder.parameters():
                param.requires_grad = False

            for layer in encoder.layers[-self.unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Projector a classifier sú vždy trénovateľné
        for param in self.wav2vec.projector.parameters():
            param.requires_grad = True
        for param in self.wav2vec.classifier.parameters():
            param.requires_grad = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass – očakáva surový waveform (1D alebo 2D) a vracia logits."""
        waveform = waveform.to("cuda")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Feature extractor očakáva numpy array
        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to("cuda")
        outputs = self.wav2vec(input_values=input_values, output_hidden_states=False)

        return outputs.logits