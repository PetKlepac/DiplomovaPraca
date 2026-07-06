"""
Download d2/, d4/, d5/ and testing/ from Zenodo.
Run this script from inside the b_dataset/ folder.
"""

import requests
import zipfile
import io
from pathlib import Path

# ================== CONFIGURATION ==================
ZENODO_RECORD_ID = "21222553"          # ← CHANGE THIS to your real Zenodo record ID

# Name of the ZIP file you will upload to Zenodo
ZIP_FILENAME = "b_dataset.zip"

def main():
    target_dir = Path(__file__).parent          # same folder as this script
    print(f"Downloading {ZIP_FILENAME} from Zenodo...")

    url = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{ZIP_FILENAME}?download=1"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(target_dir)

    print(f"✓ Successfully extracted to: {target_dir}")
    print("  → d2/, d4/, d5/ and testing/ are now available.")


if __name__ == "__main__":
    main()