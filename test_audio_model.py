import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)

MODEL_NAME = "Hemgg/Deepfake-audio-detection"

print("Downloading audio model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME
)

model.to("cpu")
model.eval()

print("\nModel downloaded successfully!")
print("Device: CPU")
print("Labels:", model.config.id2label)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("\nAudio model is ready!")
