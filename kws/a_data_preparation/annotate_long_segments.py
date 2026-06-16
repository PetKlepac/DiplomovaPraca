"""
Interaktívna anotácia audio súborov klávesnicou.

Vstup:
    Priečinok s pripravenými 1-sekundovými segmentmi
    (temporary_segments_celyden1)

Výstup:
    Segmenty roztriedené do podpriečinkov podľa anotácie:
      - positive/       → pozitívne vzorky
      - negative/       → negatívne vzorky
      - engpositive/    → anglické pozitívne vzorky
      - bin/       → odpad (zlé, šumové, nesprávne orezané)

Klávesy sa generujú dynamicky z konfigurácie CATEGORIES, ale predvolene sú:
  0 → preskočiť
  1 → negatívne (negative)
  3 → bin (odpad)
  7 → pozitívne (positive)
  9 → anglické pozitívne (engpositive)
  p → prehrať / zastaviť
  r → prehrať znova
  q → ukončiť

So súčasným nastavením potrebuje anotátor po skončení presunúť podpriečinky "positive" a "negative" do priečinku
testing. Následne môže byť priečinok temporary s jeho obsahom zmazaný, ak je to vhodné.
"""

import sys
from pathlib import Path
import pygame
import time
import shutil

# ==================== KONFIGURÁCIA ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FOLDER = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "temporary" / "long_segments_celyden1"
OUTPUT_FOLDER = PROJECT_ROOT / "kws" / "a_data_preparation" / "prepared_data" / "temporary" / "annotated_long_segments_celyden1"

# Mapovanie kláves → (priečinok prefix, popis)
CATEGORIES = {
    '0': (None, "preskočiť"),
    '1': ("negative", "negatívne"),
    '3': ("bin", "odpad / nevyužiteľné"),
    '7': ("positive", "pozitívne"),
    '9': ("engpositive", "anglické pozitívne"),
}

SUPPORTED_EXT = {".wav", ".mp3"}


def next_free_number(folder: Path) -> str:
    """Vráti nasledujúce 4-miestne číslo pre nový súbor."""
    if not folder.exists():
        return "0001"

    numbers = []
    for f in folder.glob("*.*"):
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        try:
            # Očakávaný formát: prefix_speaker_XXXX.wav
            num = int(f.stem.rsplit("_", 1)[-1])
            numbers.append(num)
        except (ValueError, IndexError):
            continue

    return f"{max(numbers, default=0) + 1:04d}"


def safe_stop_and_unload():
    """Bezpečne zastaví a uvoľní audio súbor (dôležité pre Windows)."""
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    time.sleep(0.08)  # Malý odklad výrazne pomáha na Windows


def play_audio(path: Path):
    """Prehrá audio súbor."""
    try:
        safe_stop_and_unload()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        print(f" ▶ {path.name}")
    except Exception as e:
        print(f" CHYBA prehrávania: {e}")


def main():
    pygame.mixer.init(frequency=16000, size=-16, channels=2, buffer=512)
    pygame.mixer.music.set_volume(0.92)

    source = INPUT_FOLDER.resolve()
    if not source.is_dir():
        print(f"Neexistuje zdrojový priečinok: {source}")
        sys.exit(1)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Zoznam audio súborov
    audio_files = sorted(
        (f for f in source.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXT),
        key=lambda p: p.name.lower()
    )

    if not audio_files:
        print("Žiadne audio súbory na anotáciu.")
        return

    print(f"Nájdených {len(audio_files)} súborov na anotáciu")
    print("────────────────────────────────────────────────────────────")
    print(" Klávesy:")
    for k in sorted(CATEGORIES.keys(), key=lambda x: (x.isdigit() and int(x), x)):
        _, desc = CATEGORIES[k]
        print(f" {k} → {desc}")
    print(" p → prehrať / zastaviť")
    print(" r → prehrať znova")
    print(" q → ukončiť")
    print("────────────────────────────────────────────────────────────\n")

    idx = 0
    while idx < len(audio_files):
        current = audio_files[idx]
        print(f"[{idx + 1:3d}/{len(audio_files):3d}] {current.name}")

        play_audio(current)

        while True:
            cmd = input("→ ").strip().lower()

            if cmd in ("q", ""):
                print("\nKoniec anotácie.")
                return

            elif cmd == "p":
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    print(" ■ stopped")
                else:
                    play_audio(current)
                continue

            elif cmd == "r":
                play_audio(current)
                continue

            elif cmd in CATEGORIES:
                prefix, desc = CATEGORIES[cmd]

                if prefix is None:
                    print(" Preskočené")
                    idx += 1
                    break

                target_folder = OUTPUT_FOLDER / prefix
                target_folder.mkdir(exist_ok=True)

                number = next_free_number(target_folder)
                stem = current.stem
                speaker = stem.split("_seg_")[0] if "_seg_" in stem else stem
                new_name = f"{prefix}_{speaker}_{number}{current.suffix}"
                new_path = target_folder / new_name

                # === HLAVNÁ OPRAVA ===
                safe_stop_and_unload()

                try:
                    # Používame copy + delete - spoľahlivejšie na Windows
                    shutil.copy2(current, new_path)
                    current.unlink()
                    print(f" → {new_name} ({desc})")
                    idx += 1
                    break
                except Exception as e:
                    print(f" CHYBA pri presune: {e}")
                    # Retry
                    time.sleep(0.3)
                    try:
                        shutil.copy2(current, new_path)
                        current.unlink()
                        print(f" → {new_name} (po retry)")
                        idx += 1
                        break
                    except Exception as e2:
                        print(f" Druhý pokus zlyhal: {e2}")
                        input("Stlač Enter pre pokračovanie...")
                break

            else:
                print(" Neplatný príkaz (0,1,3,7,9,p,r,q)")

    print("\nAnotácia dokončená!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrerušenie (Ctrl+C)")
    except Exception as e:
        print(f"Neočakávaná chyba: {e}")
    finally:
        pygame.mixer.quit()