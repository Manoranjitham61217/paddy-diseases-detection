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


def is_paddy_leaf(img):
    img = cv2.resize(img, (224, 224))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    if green_ratio < 0.25:
        return False

    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return False

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = h / (w + 1e-5)

    if area < 2000:
        return False

    if aspect_ratio < 1.8:
        return False

    return True

