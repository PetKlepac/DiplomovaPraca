from pathlib import Path
import soundfile as sf
import torch
import torchaudio.transforms as T
from collections import defaultdict
from typing import Tuple, List, Optional


# ────────────────────────────────────────────────────────────────
# Helper functions for listing files and reading labels
# ────────────────────────────────────────────────────────────────

def list_audio_files(testing_set_dir: Path, audio_extensions: set) -> list[Path]:
    """Vráti zoradený zoznam audio súborov."""
    audio_files = sorted([
        p for p in testing_set_dir.iterdir()
        if p.is_file() and p.suffix.lower() in audio_extensions
    ])

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {testing_set_dir}")

    return audio_files


def get_status_from_txt_by_file_id(txt_path: Path, file_id: str) -> str:
    """Vráti status (detected/partial/none/confuse) podľa file_id."""
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    def _norm_id(value: str) -> str:
        value = str(value).strip()
        return str(int(value)) if value.isdigit() else value

    wanted_id = _norm_id(file_id)

    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            row_id = _norm_id(parts[0])
            if row_id == wanted_id:
                return parts[3].strip().lower()

    raise KeyError(f"file_id '{file_id}' not found in {txt_path.name}")


def get_sentence_info_from_txt_by_file_id(txt_path: Path, file_id: str) -> Tuple[str, str]:
    """Vráti (text_vety, status) podľa file_id."""
    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    def _norm_id(value: str) -> str:
        value = str(value).strip()
        return str(int(value)) if value.isdigit() else value

    wanted_id = _norm_id(file_id)

    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            row_id = _norm_id(parts[0])
            if row_id == wanted_id:
                sentence_text = parts[1].strip()
                status = parts[3].strip().lower()
                return sentence_text, status

    raise KeyError(f"file_id '{file_id}' not found in {txt_path.name}")


# ────────────────────────────────────────────────────────────────
# Wav2Vec2 specific inference functions
# ────────────────────────────────────────────────────────────────

def infer_positive_prob(
        model: torch.nn.Module,
        chunk_waveform: torch.Tensor,  # raw 1D waveform: (time,)
        sample_rate: int = 16000,
) -> float:
    """
    Performs inference on a single chunk using Wav2VecKWS model.
    Returns probability of the positive class (index 1).
    """
    # Force mono if somehow stereo slipped in
    if chunk_waveform.ndim == 2:
        chunk_waveform = chunk_waveform.mean(dim=1)

    # Add batch dimension: (time,) -> (1, time)
    if chunk_waveform.ndim == 1:
        chunk_waveform = chunk_waveform.unsqueeze(0)

    chunk_waveform = chunk_waveform.float()

    with torch.no_grad():
        logits = model(chunk_waveform)  # model internally calls feature_extractor

        # Handle case when model returns tuple (some HF models do)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        prob_positive = probs[1].item()  # positive class = index 1

    return float(prob_positive)


def sentence_probability(
        model: torch.nn.Module,
        audio_path: Path,
        sample_rate: int = 16000,
        window_length_sec: float = 1.0,
        window_hop_sec: float = 0.1,
        aggregation_mode: str = "max",
        return_all_probs: bool = False,
) -> float | tuple[float, list[float]]:
    """
    Computes detection probability for entire audio file using sliding window.
    Designed specifically for Wav2VecKWS model (raw waveform input).
    """
    # Load audio
    waveform_np, orig_sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(waveform_np).float()

    # Force mono if stereo
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)  # (time, channels) -> (time,)

    # Resample if necessary
    if orig_sr != sample_rate:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    # Sliding window parameters
    win = int(window_length_sec * sample_rate)
    hop = int(window_hop_sec * sample_rate)
    total = waveform.shape[0]

    # If audio is shorter than one window
    if total < win:
        prob = infer_positive_prob(model, waveform, sample_rate)
        return (prob, [prob]) if return_all_probs else prob

    # Sliding window inference
    probs: List[float] = []
    start = 0
    while start + win <= total:
        chunk = waveform[start:start + win]
        probs.append(infer_positive_prob(model, chunk, sample_rate))
        start += hop

    # Add last overlapping chunk if there is remainder
    if probs and start < total:
        last_chunk = waveform[-win:]
        probs.append(infer_positive_prob(model, last_chunk, sample_rate))

    # Aggregation
    if not probs:
        final_prob = 0.0
    elif aggregation_mode.lower() == "mean":
        final_prob = sum(probs) / len(probs)
    else:  # default = "max"
        final_prob = max(probs)

    if return_all_probs:
        return float(final_prob), probs
    return float(final_prob)