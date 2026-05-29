# live_kws_listener.py
from pathlib import Path
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import sys
import time

import keyboard

from kws.c_model.m1_basic import BasicKWSCNN
from kws.h_testing.test_utils import sentence_probability

# ────────────────────────────────────────────────────────────────
# PARAMETRE – rovnaké ako v testovacom súbore
# ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_LENGTH_SEC = 1.0
WINDOW_HOP_SEC = 0.2
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2

SENTENCE_THRESHOLD = 0.5
AGGREGATION_MODE = "max"

# Cesta k modelu
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "kws" / "g_result" / "d6_m1_basic_c" / "d6_m1_basic_c.pth"

# ────────────────────────────────────────────────────────────────
# NAČÍTANIE MODELU
# ────────────────────────────────────────────────────────────────
print("Načítavam model...", file=sys.stderr)
model = BasicKWSCNN(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print("Model načítaný.\n", file=sys.stderr)

# ────────────────────────────────────────────────────────────────
# NAHRÁVANIE + PREHRÁVANIE
# ────────────────────────────────────────────────────────────────
def record_audio():
    print("\nStlač ENTER → začať nahrávať", file=sys.stderr)
    keyboard.wait('enter')

    print("🎤 Nahrávam... (stlač ENTER → zastaviť)", file=sys.stderr)

    recording = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    stream.start()

    try:
        while True:
            chunk, _ = stream.read(1024)
            recording.append(chunk)
            time.sleep(0.01)

            if keyboard.is_pressed('enter'):
                time.sleep(0.15)
                break
    finally:
        stream.stop()
        stream.close()

    audio = np.concatenate(recording, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    print(f"Nahrané {duration:.2f} sekúnd", file=sys.stderr)

    # === PREHRÁVANIE toho, čo si povedal ===
    print("Prehrávam nahrávku...", file=sys.stderr)
    sd.play(audio, SAMPLE_RATE)
    sd.wait()                                   # počká, kým sa dohrá

    return audio


# ────────────────────────────────────────────────────────────────
# HLAVNÝ LOOP
# ────────────────────────────────────────────────────────────────
print("\n=== Live KWS Listener (s prehrávaním) ===", file=sys.stderr)
print("ENTER → spustiť nahrávanie", file=sys.stderr)
print("ENTER → zastaviť → prehrať → vyhodnotiť\n", file=sys.stderr)
print("Ctrl + C → ukončiť\n", file=sys.stderr)

temp_path = Path("temp_live_recording.wav")

try:
    while True:
        audio = record_audio()

        if len(audio) < SAMPLE_RATE * 0.5:
            print("Nahrávka je príliš krátka.", file=sys.stderr)
            continue

        sf.write(temp_path, audio, SAMPLE_RATE)

        # Sliding window vyhodnotenie (rovnaké ako v testovacom)
        prob = sentence_probability(
            model=model,
            audio_path=temp_path,
            sample_rate=SAMPLE_RATE,
            window_length_sec=WINDOW_LENGTH_SEC,
            window_hop_sec=WINDOW_HOP_SEC,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            aggregation_mode=AGGREGATION_MODE,
        )

        # Výstup
        if prob > SENTENCE_THRESHOLD:
            print(f"{prob:.4f}  → DETEKCIA")
        else:
            print(f"{prob:.4f}  (nízka)")

        if temp_path.exists():
            temp_path.unlink()

except KeyboardInterrupt:
    print("\nUkončené.", file=sys.stderr)
finally:
    if temp_path.exists():
        temp_path.unlink()

