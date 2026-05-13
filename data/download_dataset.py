"""
Download SOCOFing dataset from Kaggle.
Make sure you have your Kaggle API key set up:
  1. Go to https://www.kaggle.com/account
  2. Click "Create New API Token" → downloads kaggle.json
  3. Place it at ~/.kaggle/kaggle.json  (Linux/Mac)
               or C:/Users/<user>/.kaggle/kaggle.json (Windows)
"""

import os
import subprocess
import zipfile

DATASET = "ruizgara/socofing"
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("📥 Downloading SOCOFing dataset from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DOWNLOAD_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Kaggle download failed:")
        print(result.stderr)
        print("\n👉 Make sure kaggle is installed: pip install kaggle")
        print("   And your API key is at ~/.kaggle/kaggle.json")
        return False

    # Unzip
    zip_path = os.path.join(DOWNLOAD_DIR, "socofing.zip")
    if os.path.exists(zip_path):
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DOWNLOAD_DIR)
        os.remove(zip_path)

    print(f"✅ Dataset ready at: {DOWNLOAD_DIR}")
    return True


if __name__ == "__main__":
    download()
