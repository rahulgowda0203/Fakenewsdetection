import torch
import soundfile as sf
import numpy as np

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)
from scipy.signal import resample_poly
from math import gcd

MODEL_NAME = "Hemgg/Deepfake-audio-detection"

print("Loading audio detector...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.to("cpu")
model.eval()


def detect_audio(file_path):
    audio, sample_rate = sf.read(file_path)

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample audio to 16 kHz
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor
        )
        sample_rate = 16000

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_index = torch.argmax(probabilities).item()

    label = model.config.id2label[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "probabilities": {
            model.config.id2label[i]: round(
                probabilities[i].item() * 100, 2
            )
            for i in range(len(probabilities))
        }
    }


print("Audio detector ready!")
