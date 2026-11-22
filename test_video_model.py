from transformers import AutoConfig

MODEL_NAME = "Hemg/Deepfake-Detection"

print("Loading video model configuration...")

config = AutoConfig.from_pretrained(MODEL_NAME)

print("\nArchitecture:", config.architectures)
print("Labels:", config.id2label)
print("\nVideo model configuration downloaded successfully!")
