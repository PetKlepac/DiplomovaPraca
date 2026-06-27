"""
Skript na analýzu efektivity modelu Wav2VecKWS.

Tento skript vykonáva:
- Načítanie natrénovaného modelu Wav2VecKWS
- Výpočet celkového počtu parametrov a počtu trénovateľných parametrov
- Benchmark inferencie (priemerný čas spracovania 1-sekundového segmentu)
- Výpočet Real-Time Factor (RTF)
- Meranie maximálnej spotreby GPU pamäte (peak memory)

Výstup: Výpis informácií do konzoly
"""

from pathlib import Path
import torch
import time
from kws.c_model.m_w2v import Wav2VecKWS
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# ────────────────────────────────────────────────────────────────
# KONFIGURÁCIA
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "d2_m_w2v_c"
EPOCH_TO_USE = 10

WINDOW_SECONDS = 1.0
SAMPLE_RATE = 16000

NUM_WARMUP = 10
NUM_RUNS = 100

# ────────────────────────────────────────────────────────────────
# CESTY
# ────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]

model_dir = project_root / "kws" / "g_result" / MODEL_NAME / "models"
model_path = model_dir / f"{MODEL_NAME}_e{EPOCH_TO_USE:02d}.pth"

# ────────────────────────────────────────────────────────────────
# KONTROL A VÝPIS
# ────────────────────────────────────────────────────────────────
if not model_path.exists():
    print(f"Model nenájdený: {model_path}")
    print("Dostupné modely:")
    for p in sorted(model_dir.glob("*.pth")):
        print(f"  {p.name}")
    raise FileNotFoundError(f"Model {model_path.name} neexistuje!")

print(f"Model: {model_path.name}")
print(f"Cesta k modelu: {model_path}\n")

# ────────────────────────────────────────────────────────────────
# NAČÍTANIE MODELU
# ────────────────────────────────────────────────────────────────
print("Načítavam Wav2VecKWS model...")
model = Wav2VecKWS(num_classes=2)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model načítaný na device: {device}\n")

# ────────────────────────────────────────────────────────────────
# FUNKCIE
# ────────────────────────────────────────────────────────────────
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def benchmark_inference(model, device, num_warmup=10, num_runs=100, segment_length=16000):
    """Zmeria priemerný čas inferencie a peak GPU memory."""
    dummy_input = torch.randn(segment_length).to(device)

    # Reset peak memory stats (len na CUDA)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Meranie času
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) / num_runs * 1000
    rtf = avg_time_ms / 1000.0

    # Peak memory (len na CUDA)
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return avg_time_ms, rtf, peak_memory_mb


# ────────────────────────────────────────────────────────────────
# HLAVNÁ ANALÝZA
# ────────────────────────────────────────────────────────────────
print("=" * 70)
print("ANALÝZA MODELU")
print("=" * 70)

# 1. Počet parametrov
total_params, trainable_params = count_parameters(model)
model_size_mb = model_path.stat().st_size / (1024 * 1024)

print(f"\n[PARAMETRE]")
print(f"Celkový počet parametrov:      {total_params:,}")
print(f"Trénovateľné parametre:        {trainable_params:,}")
print(f"Percento trénovateľných:       {100 * trainable_params / total_params:.2f} %")
print(f"Veľkosť modelu na disku:       {model_size_mb:.2f} MB")

# 2. Benchmark inferencie + Peak Memory
print(f"\n[BENCHMARK INFERENCIE + GPU MEMORY]")
avg_time_ms, rtf, peak_memory_mb = benchmark_inference(
    model, device,
    num_warmup=NUM_WARMUP,
    num_runs=NUM_RUNS,
    segment_length=int(WINDOW_SECONDS * SAMPLE_RATE)
)

print(f"Priemerný čas na 1 segment:    {avg_time_ms:.2f} ms")
print(f"Real-Time Factor (RTF):        {rtf:.4f}")

if device.type == "cuda":
    print(f"Peak GPU Memory (inferencia): {peak_memory_mb:.2f} MB")
else:
    print("Peak GPU Memory:             N/A (beží na CPU)")

print("\n" + "=" * 70)
print("Hotovo.")
print("=" * 70)