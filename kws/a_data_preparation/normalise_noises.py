"""
Skript na normalizáciu a štandardizáciu súborov so šumom.

Účel:
    Spracuje všetky audio súbory zo vstupného priečinka a vytvorí normalizované
    verzie s cieľovou hlasitosťou -16 dBFS. Súbory sa premenujú na n1.wav, n2.wav, ...
    a orežú na maximálne 10 sekúnd.
"""

from pathlib import Path
from pydub import AudioSegment


# ==================== KONFIGURÁCIA ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "raw_data" / "unprepared_noises"
output_folder = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "training" / "noises"

TARGET_DB = -16  # Cieľová hlasitosť šumu v dBFS


def main():
    # Vytvorenie výstupného priečinka
    output_folder.mkdir(parents=True, exist_ok=True)

    # Podporované formáty
    audio_extensions = ['*.wav', '*.mp3']

    # Načítanie všetkých súborov
    all_noise_files = []
    for ext in audio_extensions:
        all_noise_files.extend(input_folder.glob(ext))

    print(f"Nájdených {len(all_noise_files)} šumových súborov v priečinku:")
    print(f"{input_folder}\n")
    print(f"Cieľová hlasitosť šumu: {TARGET_DB} dBFS\n")

    # ==================== SPRACOVANIE ====================
    for idx, file_path in enumerate(all_noise_files, start=1):
        try:
            print(f"Spracovávam ({idx}/{len(all_noise_files)}): {file_path.name} → n{idx}.wav")

            # Načítanie súboru
            audio = AudioSegment.from_file(file_path)
            original_duration = len(audio) / 1000.0

            # Orezať na maximálne 10 sekúnd
            if len(audio) > 10000:
                trimmed_audio = audio[:10000]
                print(f" → Orezať z {original_duration:.1f}s na 10.0s")
            else:
                trimmed_audio = audio
                print(f" → Ponechaná originálna dĺžka ({original_duration:.1f}s)")

            # Normalizácia
            normalized_audio = trimmed_audio.normalize()
            gain_needed = TARGET_DB - normalized_audio.dBFS
            final_audio = normalized_audio.apply_gain(gain_needed)

            # Uloženie
            output_filename = f"n{idx}.wav"
            output_path = output_folder / output_filename

            final_audio.export(output_path, format="wav")
            print(f" ✓ Hotovo: {output_filename} | {final_audio.dBFS:.1f}dB\n")

        except Exception as e:
            print(f" ✗ Chyba pri {file_path.name}: {e}\n")

    # ==================== ZÁVEREČNÁ SPRÁVA ====================
    print("\n" + "=" * 80)
    print("HOTOVO!")
    print(f"Počet spracovaných súborov: {len(all_noise_files)}")
    print(f"Výstupný priečinok: {output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()