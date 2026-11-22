
from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    BASE / "datasets/video/SynthForensics_sample/I2V/videos/canonical"
)

OUTPUT_DIR = BASE / "video_training/extracted_frames"

FRAMES_PER_VIDEO = 10
TARGET_SIZE = (224, 224)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_paths = sorted(VIDEO_DIR.rglob("*.mp4"))

print("Videos found:", len(video_paths))

total_frames = 0
failed_videos = 0

for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path.name)
        failed_videos += 1
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        failed_videos += 1
        continue

    video_name = video_path.stem
    video_output = OUTPUT_DIR / video_name
    video_output.mkdir(parents=True, exist_ok=True)

    positions = [
        int(i * (frame_count - 1) / (FRAMES_PER_VIDEO - 1))
        for i in range(FRAMES_PER_VIDEO)
    ]

    for frame_number, position in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(frame, TARGET_SIZE)
        output_path = video_output / f"frame_{frame_number}.jpg"

        cv2.imwrite(str(output_path), frame)
        total_frames += 1

    cap.release()

print("\nExtraction complete")
print("Frames extracted:", total_frames)
print("Failed videos:", failed_videos)
print("Output folder:", OUTPUT_DIR)
