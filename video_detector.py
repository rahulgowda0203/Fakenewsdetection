import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "Hemg/Deepfake-Detection"

print("Loading video frame detector...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.to("cpu")
model.eval()

print("Video frame detector ready!")


def detect_video(file_path, max_frames=10):

    video = cv2.VideoCapture(file_path)

    if not video.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        video.release()
        raise ValueError("Video contains no readable frames.")

    frame_positions = [
        int(i * (total_frames - 1) / max(1, max_frames - 1))
        for i in range(min(max_frames, total_frames))
    ]

    results = []

    for frame_number in frame_positions:

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_index = torch.argmax(probabilities).item()

        label = model.config.id2label[predicted_index]
        confidence = probabilities[predicted_index].item() * 100

        results.append({
            "frame": frame_number,
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    video.release()

    if not results:
        raise ValueError("No video frames could be analysed.")

    fake_count = sum(
        1 for result in results
        if result["prediction"].lower() == "fake"
    )

    fake_percentage = (fake_count / len(results)) * 100

    return {
        "total_frames_analyzed": len(results),
        "fake_frames": fake_count,
        "real_frames": len(results) - fake_count,
        "fake_percentage": round(fake_percentage, 2),
        "frame_results": results,
        "overall_result": (
            "Potential manipulation detected"
            if fake_percentage >= 50
            else "No strong manipulation signal detected"
        )
    }
