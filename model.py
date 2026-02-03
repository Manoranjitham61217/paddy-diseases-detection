import numpy as np
import cv2
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("rice_model.keras")

CLASS_NAMES = ["BacterialBlight", "Blast", "BrownSpot"]

IMG_SIZE = 224


def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def generate_gradcam(img_array, class_index):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-9

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    return heatmap


def predict_disease_with_gradcam(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = preprocess_image(img)

    preds = model.predict(img_array)[0]
    confidence = float(np.max(preds))
    entropy = float(-np.sum(preds * np.log(preds + 1e-9)))

    # -------- Soft validation (IMPORTANT) --------
    is_uncertain = False

    if confidence < 0.20:
        is_uncertain = True

    if entropy < 0.05:
        is_uncertain = True

    # Geometry warning (NOT rejection)
    geometry_warning = False
    h, w, _ = img_resized.shape
    if h / (w + 1e-5) < 1.0:
        geometry_warning = True

    if is_uncertain:
        return {
            "status": "not_paddy_leaf",
            "confidence": confidence,
            "entropy": entropy,
        }

    class_index = np.argmax(preds)
    disease = CLASS_NAMES[class_index]

    heatmap = generate_gradcam(img_array, class_index)

    return {
        "status": "confident",
        "prediction": disease,
        "confidence": confidence,
        "entropy": entropy,
        "geometry_warning": geometry_warning,
        "gradcam": heatmap,
    }
