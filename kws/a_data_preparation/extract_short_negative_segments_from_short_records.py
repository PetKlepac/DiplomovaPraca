"""
Skript na extrakciu rovnomerne rozložených segmentov z JEDNÉHO konkrétneho audio súboru.

Účel:
    Z dlhšieho audio súboru vygeneruje rovnomerne rozložené 1-sekundové segmenty
    s malou náhodnou odchýlkou. Následne umožňuje interaktívne schvaľovanie
    (počúvanie + rozhodnutie uložiť/zahodiť).
"""

import os
import random
import tempfile
from pathlib import Path

from pydub import AudioSegment
import winsound


# ==================== KONFIGURÁCIA ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_file = PROJECT_ROOT / "kws" / "a_data_preparation" / "raw_data" / "short_records" / "nedele.wav"
output_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "negative" / "airport_negative_nedele_500"

BASE_NAME = "nedele_seg"
NUM_SEGMENTS_TO_GENERATE = 530
SEGMENT_LENGTH_SEC = 1.0
TARGET_SR = 16000
RANDOM_SEED = 42


def play_audio_segment(audio_segment: AudioSegment, display_name: str):
    """Prehrá segment a čaká na rozhodnutie používateľa."""
    print(f"\n▶ Prehrávam: {display_name}")
    print(" [Enter] = Uložiť | [n] = Zahodiť | [r] = Opakovať | [q] = Ukončiť")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
        audio_segment.export(temp_path, format="wav")

        try:
            winsound.PlaySound(temp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f" Chyba prehrávania: {e}")

        while True:
            choice = input(" Voľba (Enter/n/r/q): ").strip().lower()

            if choice in ["", "y", "yes", "ano"]:
                os.unlink(temp_path)
                return True
            elif choice == "n":
                os.unlink(temp_path)
                return False
            elif choice == "r":
                os.unlink(temp_path)
                return play_audio_segment(audio_segment, display_name)
            elif choice == "q":
                os.unlink(temp_path)
                return None
            else:
                print(" Zadaj: Enter / n / r / q")


def main():
    random.seed(RANDOM_SEED)

    input_file_path = Path(input_file)
    if not input_file_path.exists():
        print(f" Súbor nebol nájdený: {input_file_path}")
        return

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f" Načítavam súbor: {input_file_path.name}")

    try:
        audio = AudioSegment.from_file(input_file_path)

        # Konverzia na mono a požadovanú vzorkovaciu frekvenciu
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != TARGET_SR:
            audio = audio.set_frame_rate(TARGET_SR)

        total_ms = len(audio)
        segment_length_ms = int(SEGMENT_LENGTH_SEC * 1000)

        if total_ms < segment_length_ms:
            print(" Nahrávka je príliš krátka!")
            return

        # Generovanie kandidátov
        candidates = []
        print(f"Generujem {NUM_SEGMENTS_TO_GENERATE} rovnomerne rozložených segmentov...\n")

        available_length = total_ms - segment_length_ms
        step = available_length / max(1, NUM_SEGMENTS_TO_GENERATE)

        for i in range(NUM_SEGMENTS_TO_GENERATE):
            # Rovnomerné rozloženie + náhodná odchýlka
            center = i * step + step / 2
            jitter = random.uniform(-step * 0.35, step * 0.35)
            start_ms = int(center + jitter)
            start_ms = max(0, min(start_ms, total_ms - segment_length_ms))

            segment = audio[start_ms : start_ms + segment_length_ms]
            candidates.append((segment, f"start{start_ms}ms"))

            print(f" ✓ {i+1:03d} | start = {start_ms:6d} ms")

        print(f"\n=== VYGENEROVANÝCH {len(candidates)} SEGMENTOV ===")
        print("Začína interaktívne schvaľovanie...\n")

        saved_count = 0
        for i, (seg, origin) in enumerate(candidates, 1):
            decision = play_audio_segment(seg, f"{i:03d} - {origin}")

            if decision is None:        # q = quit
                break

            if decision:
                saved_count += 1
                output_file = output_path / f"{BASE_NAME}_{saved_count:03d}.wav"
                seg.export(output_file, format="wav")
                print(f" ULOŽENÉ → {output_file.name}\n")
            else:
                print(" 🗑 Zahodené\n")

    except Exception as e:
        print(f"Chyba pri spracovaní súboru: {e}")

    print("\n" + "="*80)
    print(f"HOTOVO! Uložených {saved_count} segmentov.")
    print(f"Adresár: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()