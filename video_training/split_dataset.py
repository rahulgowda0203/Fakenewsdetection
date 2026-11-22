
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MANIFEST = BASE / "video_training/video_manifest.csv"

df = pd.read_csv(MANIFEST)

# Hold out two complete source-image IDs.
test_ids = ["621", "717"]

df["split"] = df["source_id"].astype(str).apply(
    lambda x: "test" if x in test_ids else "train"
)

output = BASE / "video_training/video_manifest_split.csv"
df.to_csv(output, index=False)

print("Split saved:", output)
print("\nSplit summary:")
print(df.groupby(["split", "generator"]).size().unstack(fill_value=0))

print("\nUnique source IDs:")
print("Training:", sorted(df[df["split"] == "train"]["source_id"].unique()))
print("Testing:", sorted(df[df["split"] == "test"]["source_id"].unique()))

assert set(df[df["split"] == "train"]["source_id"]).isdisjoint(
    set(df[df["split"] == "test"]["source_id"])
)
assert len(df) == 700

