
from pathlib import Path
import tensorflow as tf
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "models" / "image" / "image_authenticity.keras"
)

model = None

def load_image_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Image model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Image authenticity model loaded successfully!")


def predict_image(image_path):
    global model

    if model is None:
        load_image_model()

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    real_probability = float(predictions[0])
    fake_probability = float(predictions[1])

    predicted_class = int(np.argmax(predictions))

    if predicted_class == 0:
        label = "Likely Real Image"
    else:
        label = "Potentially AI-Generated Image"

    return {
        "prediction": label,
        "real_probability": round(real_probability * 100, 2),
        "fake_probability": round(fake_probability * 100, 2),
        "confidence": round(
            float(np.max(predictions)) * 100, 2
        )
    }
