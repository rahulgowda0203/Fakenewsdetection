import torch
from transformers import AutoModelForImageClassification

MODEL_NAME = "Hemg/Deepfake-Detection"

print("Downloading video frame detection model...")

model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.to("cpu")
model.eval()

print("\nModel downloaded successfully!")
print("Architecture:", model.config.architectures)
print("Labels:", model.config.id2label)
print("Parameters:", sum(p.numel() for p in model.parameters()))

del model
