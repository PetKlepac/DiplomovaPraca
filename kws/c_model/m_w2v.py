"""Wav2VecKWS - Optimalizovaná implementácia wav2vec2 pre Keyword Spotting (KWS)."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class Wav2VecKWS(nn.Module):
    """wav2vec2 model prispôsobený na detekciu kľúčových slov."""

    def __init__(
        self,
        num_classes: int = 2,           # počet tried na klasifikáciu
        freeze_feature_extractor: bool = True,  # zamrazí CNN feature extractor
        freeze_encoder: bool = True,    # zamrazí Transformer encoder
    ):
        super().__init__()
        self.model_name = "fav-kky/wav2vec2-base-cs-80k-ClTRUS"  # názov predtrénovaného modelu z Hugging Face
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)  # načítanie feature extractora
        self.wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,         # nastaví výstupnú vrstvu na požadovaný počet tried
            problem_type="single_label_classification",# definuje typ úlohy (softmax + cross entropy)
            ignore_mismatched_sizes=True,   # umožní nahratie modelu s iným počtom tried
        ).to('cuda')
        self.freeze_feature_extractor = freeze_feature_extractor  # uloženie nastavenia zamrznutia
        self.freeze_encoder = freeze_encoder                      # uloženie nastavenia zamrznutia
        self.apply_freezing()
        print(f"✓ Model {self.model_name} úspešne načítaný na GPU")

    def apply_freezing(self):
        """Zamrazí vybrané časti modelu."""
        if self.freeze_feature_extractor:
            for param in self.wav2vec.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            print("✓ Feature extractor (CNN) zmrazený")
        if self.freeze_encoder:
            for param in self.wav2vec.wav2vec2.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder zmrazený (trénuje sa iba hlava)")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass - prijíma raw waveform."""
        waveform = waveform.to('cuda')  # presun na GPU
        inputs = self.feature_extractor(
            waveform.cpu().numpy(),         # prevod na numpy (transformers očakáva CPU numpy)
            sampling_rate=16000,            # wav2vec2 je natrénovaný na 16kHz
            return_tensors="pt",            # vráti PyTorch tensory
            padding=True,                   # automatické padding na najdlhší vzorok v batchi
            # return_attention_mask=True,
        )
        input_values = inputs.input_values.to('cuda')  # hlavný vstup na GPU
        # attention_mask = inputs.attention_mask.to('cuda')
        outputs = self.wav2vec(
            input_values=input_values,      # hlavný vstup pre wav2vec2
            # attention_mask=attention_mask,
            output_hidden_states=False,     # nevracia hidden states (šetrí pamäť)
        )
        return outputs.logits