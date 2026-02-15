import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Ensure these match your folder names exactly (Alphabetical order is standard)
CLASS_NAMES = ["BacterialBlight", "Blast", "BrownSpot", "Non_Paddy"]

# Detailed Treatment and Suggestions
DISEASE_INFO = {
    "BacterialBlight": {
        "treatment": "Apply balanced fertilizer (avoid excess Nitrogen). Use copper-based fungicides like Copper Hydroxide if infection is severe.",
        "prevention": "Ensure proper field drainage and use disease-free seeds."
    },
    "Blast": {
        "treatment": "Avoid high nitrogen application. Spray fungicides such as Tricyclazole or Carbendazim.",
        "prevention": "Remove weeds that act as alternative hosts and use resistant varieties."
    },
    "BrownSpot": {
        "treatment": "Correct soil nutrient deficiencies (specifically Potassium). Apply fungicides like Mancozeb.",
        "prevention": "Ensure proper irrigation and improve soil health."
    }
}

def load_trained_model(model_path="rice_model.keras"):
    """Loads the MobileNetV2 model saved from your notebook."""
    return tf.keras.models.load_model(model_path)

def predict_image(model, image):
    """Preprocesses and predicts the class of the input image."""
    # ADDED: Convert image to RGB to handle PNG transparency (fixes the shape error)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    confidence = float(np.max(preds))
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]

    return label, confidence