
from pathlib import Path
from huggingface_hub import hf_hub_download
import zipfile

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset destination
DATASET_DIR = BASE_DIR / "datasets" / "image" / "authenticity"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "yonsei-dli/CAS3120-2026-Project1_AI-Generated-Image-Detection"
FILE_NAME = "processed.zip"

print("=" * 55)
print("FAKE NEWS DETECTION - IMAGE DATASET DOWNLOADER")
print("=" * 55)

try:
    # Step 1: Download dataset
    print("\n[1/3] Downloading dataset from Hugging Face...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILE_NAME,
        repo_type="dataset"
    )

    print("Download completed:", zip_path)

    # Step 2: Extract dataset
    print("\n[2/3] Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        for member in zip_ref.infolist():
            target = (DATASET_DIR / member.filename).resolve()

            # Prevent unsafe ZIP extraction
            if not target.is_relative_to(DATASET_DIR.resolve()):
                raise ValueError(
                    f"Unsafe file path detected: {member.filename}"
                )

        zip_ref.extractall(DATASET_DIR)

    # Step 3: Check downloaded files
    print("\n[3/3] Checking downloaded files...")

    for split in ["train", "val", "test"]:
        folder = DATASET_DIR / "processed" / split

        if folder.exists():
            count = sum(
                1 for p in folder.rglob("*") if p.is_file()
            )
            print(f"{split}: {count} files")
        else:
            print(f"{split}: folder not found")

    print("\nSUCCESS! Dataset downloaded and extracted.")
    print("Dataset location:", DATASET_DIR / "processed")

except Exception as error:
    print("\nDATASET DOWNLOAD FAILED")
    print("Error:", error)
    print("\nCheck your internet connection and Hugging Face access.")
