"""Wav2VecKWS - Optimalizovaná implementácia wav2vec2 pre Keyword Spotting (KWS)."""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class Wav2VecKWS(nn.Module):
    """wav2vec2 model prispôsobený na detekciu kľúčových slov."""

    def __init__(
            self,
            num_classes: int = 2,
            freeze_feature_extractor: bool = True,
            freeze_encoder: bool = True,
            unfreeze_last_n_layers: int = 3,  # ← dôležitý parameter
            verbose: bool = True,
    ):
        super().__init__()
        self.local_model_path = Path(__file__).parent / "pretrained_wav2vec"
        self.verbose = verbose

        self.freeze_feature_extractor = freeze_feature_extractor
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.local_model_path)   # Načítanie modelu

        self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.local_model_path,
            num_labels=num_classes,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        ).to('cuda')

        self._apply_freezing_strategy()

        if self.verbose:
            print(f"Feature extractor frozen: {self.freeze_feature_extractor}")
            print(f"Encoder frozen: {self.freeze_encoder} "
                  f"(unfrozen last {self.unfreeze_last_n_layers} layers)")


    def _apply_freezing_strategy(self):
        """Zamrazí časti modelu a selektívne unfreeze posledné vrstvy + head."""

        # 1. Feature extractor (CNN)
        if self.freeze_feature_extractor:
            for param in self.wav2vec.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        # 2. Encoder (Transformer)
        encoder = self.wav2vec.wav2vec2.encoder
        if self.freeze_encoder and self.unfreeze_last_n_layers > 0:
            # Freeze všetko
            for param in encoder.parameters():
                param.requires_grad = False

            # Unfreeze posledných N vrstiev
            for layer in encoder.layers[-self.unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # 3. Classification head — vždy trénujeme
        for param in self.wav2vec.projector.parameters():
            param.requires_grad = True
        for param in self.wav2vec.classifier.parameters():
            param.requires_grad = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass - prijíma raw waveform."""
        waveform = waveform.to('cuda')
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        inputs = self.feature_extractor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to('cuda')

        outputs = self.wav2vec(
            input_values=input_values,
            output_hidden_states=False,
        )
        return outputs.logits