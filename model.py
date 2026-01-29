import cv2
import numpy as np
import tensorflow as tf

# Load model once
model = tf.keras.models.load_model("rice_model.keras")

CLASS_NAMES = ["BacterialBlight", "Blast", "BrownSpot"]
CONF_THRESHOLD = 0.6

def predict_disease_with_gradcam(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    idx = int(np.argmax(preds))

    if confidence < CONF_THRESHOLD:
        return {
            "status": "uncertain",
            "confidence": confidence
        }

    return {
        "status": "confident",
        "prediction": CLASS_NAMES[idx],
        "confidence": confidence
    }
