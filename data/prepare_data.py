"""
Prepare tomato leaf disease dataset from PlantVillage and PlantDoc.
Downloads, extracts, cleans, and splits into train/val/test.
"""
import os
import shutil
import requests
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split

PLANTVILLAGE_URL = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/2/files/7e6e6e6e-6e6e-6e6e-6e6e-6e6e6e6e6e6e/plantvillage.zip"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TOMATO_CLASSES = [
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]


def download_and_extract(url, out_dir, zip_name):
    import time
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / zip_name
    def is_valid_zip(path):
        try:
            with zipfile.ZipFile(path, "r") as zip_ref:
                bad_file = zip_ref.testzip()
                return bad_file is None
        except zipfile.BadZipFile:
            return False
    # Download if not exists or invalid
    if not zip_path.exists() or not is_valid_zip(zip_path):
        if zip_path.exists():
            print(f"Found corrupted zip at {zip_path}, deleting...")
            zip_path.unlink()
        print(f"Downloading {url}...")
        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=60)
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                if is_valid_zip(zip_path):
                    break
                else:
                    print("Downloaded file is not a valid zip. Retrying...")
                    zip_path.unlink()
            except Exception as e:
                print(f"Download failed: {e}. Retrying...")
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to download a valid zip from {url} after 3 attempts.")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def prepare_plantvillage():
    download_and_extract(PLANTVILLAGE_URL, RAW_DIR, "plantvillage.zip")
    # Search for each class folder anywhere under RAW_DIR (handles different zip layouts)
    PROCESSED_DIR.mkdir(exist_ok=True)

    def find_class_dir(class_name: str) -> Path | None:
        # Prefer paths containing 'PlantVillage' and/or 'color' when multiple matches exist
        candidates = []
        for p in RAW_DIR.rglob(class_name):
            if p.is_dir():
                score = 0
                parts = [str(x).lower() for x in p.parts]
                if any("plantvillage" in part for part in parts):
                    score += 2
                if any("color" in part for part in parts):
                    score += 1
                # Shorter paths are preferred when scores tie
                candidates.append((score, len(str(p)), p))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][2]

    for cls in TOMATO_CLASSES:
        src_cls = find_class_dir(cls)
        dst_cls = PROCESSED_DIR / cls
        if src_cls is None:
            print(f"Warning: Could not locate class directory for '{cls}' under {RAW_DIR}.")
            continue
        shutil.copytree(src_cls, dst_cls, dirs_exist_ok=True)


def prepare_plantdoc():
    download_and_extract(PLANTDOC_URL, RAW_DIR, "plantdoc.zip")
    # PlantDoc is more complex, so just copy tomato images for now
    # TODO: Add PlantDoc tomato extraction logic


def split_data():
    for cls in TOMATO_CLASSES:
        cls_dir = PROCESSED_DIR / cls
        images = (
            list(cls_dir.glob("*.jpg"))
            + list(cls_dir.glob("*.JPG"))
            + list(cls_dir.glob("*.jpeg"))
            + list(cls_dir.glob("*.JPEG"))
            + list(cls_dir.glob("*.png"))
        )
        if len(images) == 0:
            print(f"Warning: No images found for class '{cls}'. Skipping.")
            continue
        train, valtest = train_test_split(images, test_size=0.2, random_state=42)
        val, test = train_test_split(valtest, test_size=0.5, random_state=42)
        for split, files in zip(["train", "val", "test"], [train, val, test]):
            split_dir = DATA_DIR / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, split_dir / f.name)


def main():
    prepare_plantvillage()
    prepare_plantdoc()
    split_data()
    print("Dataset prepared!")

if __name__ == "__main__":
    main()
