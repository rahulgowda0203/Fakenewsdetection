
from pathlib import Path
import csv

BASE = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    BASE / "datasets/video/SynthForensics_sample/I2V/videos/canonical"
)
FRAME_DIR = BASE / "video_training/extracted_frames"
OUTPUT_FILE = BASE / "video_training/video_manifest.csv"

rows = []

for video_path in sorted(VIDEO_DIR.rglob("*.mp4")):
    generator = video_path.parent.name
    video_name = video_path.stem

    frame_folder = FRAME_DIR / video_name
    source_id = video_name.split("_")[0]

    if not frame_folder.exists():
        print("Missing folder:", video_name)
        continue

    for frame_path in sorted(frame_folder.glob("*.jpg")):
        rows.append({
            "frame_path": str(frame_path.relative_to(BASE)),
            "source_id": source_id,
            "generator": generator,
            "video_name": video_name
        })

with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["frame_path", "source_id", "generator", "video_name"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Manifest created:", OUTPUT_FILE)
print("Total frames:", len(rows))

print("\nFrames per generator:")
for generator in sorted(set(row["generator"] for row in rows)):
    count = sum(row["generator"] == generator for row in rows)
    print(generator, ":", count)
