"""
Skript na segmentáciu dlhšieho audio súboru na rečové segmenty.

Účel:
    Rozdelí jeden dlhší audio súbor (nahrávku celého dňa bez filtrovania hlasných časťí) na samostatné
    rečové segmenty (vety) s bufferom na konci. Následne je ešte potrebné anotovať s annotate_long_segments.
"""

from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


# ==================== KONFIGURÁCIA ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_file = PROJECT_ROOT / "kws" / "a_data_preparation" / "raw_data" / "long_records" / "celyden1.wav"
output_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "temporary" / "long_segments_celyden1"


# Parametre segmentácie
END_BUFFER_MS = 500
MIN_SILENCE_MS = 1000
SILENCE_THRESH_DBFS = -35
MIN_LOUD_MS = 500
SEEK_STEP_MS = 10


def main():
    print(f"\nSpracovávam: {input_file.name}")

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Segmenty budú uložené do: {output_folder}\n")

    if not input_file.exists():
        print(f"Chyba: Vstupný súbor neexistuje!\n{input_file}")
        return

    # ==================== NAČÍTANIE AUDIA ====================
    audio = AudioSegment.from_wav(str(input_file))

    # ==================== DETEKCIA REČOVÝCH OBLASTÍ ====================
    loud_regions = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
        seek_step=SEEK_STEP_MS
    )

    print(f"Nájdených {len(loud_regions)} hlasných (rečových) oblastí\n")

    # ==================== VYTVÁRANIE SEGMENTOV ====================
    saved_count = 0
    base_name = input_file.stem

    for i, (start_ms, end_ms) in enumerate(loud_regions, 1):
        orig_loud_ms = end_ms - start_ms

        # Preskočiť príliš krátke časti
        if orig_loud_ms < MIN_LOUD_MS:
            print(f"Preskočené {i:03d} — príliš krátke ({orig_loud_ms/1000:.2f} s)")
            continue

        # Pridanie bufferu iba na konci
        buffered_end_ms = min(len(audio), end_ms + END_BUFFER_MS)

        # Vytvorenie segmentu
        segment = audio[start_ms:buffered_end_ms]

        # Uloženie
        segment_filename = f"{base_name}_seg_{i:03d}.wav"
        out_path = output_folder / segment_filename

        segment.export(str(out_path), format="wav")
        saved_count += 1

        print(
            f"✓ {segment_filename:<35} | "
            f"trvanie: {len(segment)/1000:5.2f} s | "
            f"pôvodná hlasná: {orig_loud_ms/1000:5.2f} s | "
            f"buffer: {(buffered_end_ms - end_ms)/1000:4.2f} s"
        )

    # ==================== ZÁVEREČNÁ SPRÁVA ====================
    print(f"\nHotovo — uložených {saved_count} segmentov")
    print(f"→ {output_folder}")


if __name__ == "__main__":
    main()