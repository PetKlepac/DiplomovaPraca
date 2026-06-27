"""
Skript na normalizáciu a štandardizáciu súborov so šumom.

Účel:
    Spracuje všetky audio súbory zo vstupného priečinka, premenujú sa na n1.wav, n2.wav, ...
    a orežú na maximálne 10 sekúnd.
"""

from pathlib import Path
from pydub import AudioSegment

# ==================== KONFIGURÁCIA ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "raw_data" / "unprepared_noises"
output_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "noises"

MAX_DURATION_MS = 10000      # maximálne 10 sekúnd

def main():
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_extensions = ['*.wav', '*.mp3', '*.ogg']
    all_noise_files = []
    for ext in audio_extensions:
        all_noise_files.extend(input_folder.glob(ext))

    print(f"Nájdených {len(all_noise_files)} šumových súborov.")

    for idx, file_path in enumerate(all_noise_files, start=1):
        print(f"Spracovávam ({idx}/{len(all_noise_files)}): {file_path.name}")
        audio = AudioSegment.from_file(file_path)

        # Orezanie
        if len(audio) > MAX_DURATION_MS:
            audio = audio[:MAX_DURATION_MS]

        # Uloženie
        output_filename = f"n{idx:03d}.wav"
        output_path = output_folder / output_filename
        audio.export(output_path, format="wav")

        print(f"  ✓ Hotovo: {output_filename} | {audio.dBFS:.2f} dBFS")


    print("\n" + "="*80)
    print("NORMALIZÁCIA ŠUMOV DOKONČENÁ")
    print(f"Počet spracovaných súborov: {len(all_noise_files)}")
    print(f"Výstup: {output_folder}")
    print("="*80)


if __name__ == "__main__":
    main()