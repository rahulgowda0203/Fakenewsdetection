from pathlib import Path

import clip
import joblib
import numpy as np
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "image" /
    "context_baseline" / "context_baseline.pkl"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "True",
    1: "Miscaptioned",
    2: "Out-of-Context"
}

# Load these once per Python process, not once per prediction.
_clip_model = None
_preprocess = None
_classifier = None


def load_detector():
    global _clip_model, _preprocess, _classifier

    if _clip_model is not None:
        return _clip_model, _preprocess, _classifier

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Context model not found: {MODEL_PATH}")

    print("Loading CLIP ViT-L/14 and image-context classifier...")

    _clip_model, _preprocess = clip.load(
        "ViT-L/14",
        device=DEVICE
    )

    _classifier = joblib.load(MODEL_PATH)

    print("Image context detector loaded successfully!")

    return _clip_model, _preprocess, _classifier


def extract_features(image_path, caption):
    model, preprocess, classifier = load_detector()

    caption = str(caption).strip()

    if not caption:
        raise ValueError("Caption cannot be empty.")

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    text_tokens = clip.tokenize(
        [caption],
        truncate=True
    ).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    image_vector = image_features.cpu().numpy().astype(np.float32)[0]
    text_vector = text_features.cpu().numpy().astype(np.float32)[0]

    similarity = float(np.dot(image_vector, text_vector))
    difference = np.abs(image_vector - text_vector)

    # Same 769-feature format used during training.
    features = np.concatenate([
        np.array([similarity], dtype=np.float32),
        difference
    ]).reshape(1, -1)

    return features, similarity, classifier


def predict(image_path, caption):
    features, similarity, classifier = extract_features(
        image_path,
        caption
    )

    prediction = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]

    probability_results = {}

    for label, probability in zip(
        classifier.classes_,
        probabilities
    ):
        class_name = CLASS_NAMES[int(label)]
        probability_results[class_name] = round(
            float(probability) * 100, 2
        )

    return {
        "prediction": CLASS_NAMES[prediction],
        "similarity": round(similarity, 4),
        "probabilities": probability_results,
        "note": (
            "Experimental classifier. This result is not proof "
            "that an image or caption is authentic."
        )
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify image-caption context consistency"
    )

    parser.add_argument("--image", required=True)
    parser.add_argument("--caption", required=True)

    args = parser.parse_args()

    result = predict(args.image, args.caption)

    print("\n========== IMAGE CONTEXT ANALYSIS ==========")
    print(f"Image: {args.image}")
    print(f"Caption: {args.caption}")
    print(f"Prediction: {result['prediction']}")
    print(f"CLIP cosine similarity: {result['similarity']}")

    print("\nClass probabilities:")
    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability:.2f}%")

    print(f"\nNote: {result['note']}")
    print("============================================")