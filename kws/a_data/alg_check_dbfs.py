"""
Skript na kontrolu priemerného loudness (dBFS) audio súborov v zadaných priečinkoch.
Používa sa na overenie úrovne hlasitosti pozitívnych (reč) a negatívnych (šum) nahrávok
pred tréningom modelu.
Podporuje .wav a .mp3 súbory.
"""

from pydub import AudioSegment
from pathlib import Path

def check_dbfs(folder_path, max_files=15):
    """
    Skontroluje priemernú hlasitosť (dBFS) audio súborov v danom priečinku.
    Zobrazí názov súboru, hodnotu dBFS a dĺžku nahrávky.
    """
    folder = Path(folder_path)

    # kontrola existence priečinka
    if not folder.exists():
        print(f"❌ Priečinok nenájdený: {folder}")
        return

    # získanie všetkých .wav a .mp3 súborov
    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
    audio_files = sorted(audio_files)[:max_files]   # obmedzenie na prvých N súborov

    # ak nie sú žiadne audio súbory
    if not audio_files:
        print(f"❌ V priečinku neboli nájdené žiadne .wav ani .mp3 súbory: {folder}")
        return

    print(f"\n📊 Kontrola dBFS v priečinku: {folder.name} ({len(audio_files)} súborov)")
    print("-" * 70)

    for file in audio_files:
        try:
            audio = AudioSegment.from_file(file)
            print(f"{file.name:45} → dBFS: {audio.dBFS:6.2f} | Trvanie: {len(audio) / 1000:.1f}s")
        except Exception as e:
            print(f"{file.name:45} → CHYBA: {e}")

    print("-" * 70)


# ====================== KONTROLA SÚBOROV ======================

print("=== REČOVÉ SÚBORY (Positive Examples - Google Chirp) ===")
check_dbfs(r"C:\Users\peter\Moje\Diplomka\DiplomovaPraca\kws\a_data\positive\1sec_hovorene_final_a_finale")
