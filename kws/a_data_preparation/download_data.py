"""
Skript na stiahnutie priečinkov prepared_data/ a raw_data/ z Zenodo.

Spusti tento skript z vnútri priečinka a_data_preparation/.

Hlavné funkcie:
- Automaticky rozpozná a rozbalí ZIP, ktorý bol vytvorený s vrchným adresárom
- Bezpečne presunie obsah na správne miesto

"""

import requests
import zipfile
import io
import shutil
from pathlib import Path
from typing import List


# ================== KONFIGURÁCIA ==================
ZENODO_RECORD_ID = "21301042"
ZIP_FILENAME = "a_data_preparation.zip"


def get_top_level_folders(z: zipfile.ZipFile) -> List[str]:
    """Vráti zoznam názvov vrchných priečinkov v ZIP archíve."""
    top_level = set()
    for name in z.namelist():
        parts = name.strip("/").split("/", 1)
        if len(parts) > 0 and parts[0]:
            top_level.add(parts[0])
    return sorted(top_level)


def main():
    target_dir = Path(__file__).parent.resolve()
    print(f"Target directory: {target_dir}\n")

    url = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{ZIP_FILENAME}?download=1"
    print(f"Downloading {ZIP_FILENAME} from Zenodo...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))
    top_level = get_top_level_folders(z)
    print(f"Top-level items in ZIP: {top_level}")

    # === Inteligentná logika rozbaľovania ===
    if len(top_level) == 1 and top_level[0] == "a_data_preparation":
        print("Detected extra top-level folder 'a_data_preparation/' → unwrapping it...")

        # Najprv rozbalíme všetko do dočasného priečinka
        temp_dir = target_dir / "_temp_extract"
        temp_dir.mkdir(exist_ok=True)
        z.extractall(temp_dir)

        # Presunieme obsah vnútorného priečinka a_data_preparation/ o úroveň vyššie
        inner_folder = temp_dir / "a_data_preparation"
        if inner_folder.exists():
            for item in inner_folder.iterdir():
                dest = target_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

            print("✓ Unwrapped successfully.")
            shutil.rmtree(temp_dir)
    else:
        # Normálne rozbalenie
        print("Extracting normally...")
        z.extractall(target_dir)

    print(f"\n✓ Extraction complete!")
    print(f" prepared_data/ and raw_data/ should now be in: {target_dir}")


if __name__ == "__main__":
    main()