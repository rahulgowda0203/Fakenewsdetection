
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets" / "image" / "verite_source" / "VERITE"

csv_path = DATA_DIR / "VERITE.csv"

df = pd.read_csv(csv_path)

print("\n========== VERITE DATASET ==========")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\n========== EMBEDDINGS ==========")

for filename in [
    "VERITE_clip_image_embeddings_ViTL14.npy",
    "VERITE_clip_text_embeddings_ViTL14.npy"
]:
    path = DATA_DIR / filename
    embeddings = np.load(path, mmap_mode="r")
    print(f"\n{filename}")
    print("Shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)

print("\n========== LABEL DISTRIBUTION ==========")

for column in df.columns:
    if any(word in column.lower() for word in ["label", "class", "truth"]):
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))
