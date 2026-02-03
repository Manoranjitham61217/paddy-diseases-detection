import streamlit as st
import cv2
import numpy as np
from model import predict_disease_with_gradcam

st.set_page_config(page_title="Paddy Disease Finder", layout="centered")

st.title("🌾 Paddy Disease Finder")
st.write("Upload a paddy (rice) leaf image to detect disease.")

uploaded_file = st.file_uploader(
    "Upload a paddy leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Selected Paddy Leaf Image", use_container_width=True)

    if st.button("🔍 Detect Disease"):
        st.write("✏️ Running leaf validation...")

        output = predict_disease_with_gradcam(img)

        # ❌ Not a paddy leaf
        if output["status"] == "not_paddy_leaf":
            st.error("❌ This is not a paddy (rice) leaf.")
            st.stop()

        # ⚠️ Geometry warning
        if output.get("geometry_warning"):
            st.warning("⚠️ Leaf appears partially cropped. Prediction may be affected.")

        # ✅ Disease result
        st.success(f"🌿 Disease Detected: {output['prediction']}")
        st.write(f"**Confidence:** {output['confidence']:.2f}")

        # Grad-CAM
        heatmap = output["gradcam"]
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_resized = cv2.resize(img, (224, 224))
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)


        st.subheader("🔍 Grad-CAM Explanation")
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        # Treatments
        st.subheader("🌱 Treatment & Improvement Suggestions")

        treatments = {
            "BacterialBlight": [
                "Use disease-resistant varieties",
                "Avoid excess nitrogen fertilizers",
                "Apply copper-based bactericides"
            ],
            "Blast": [
                "Apply fungicides like Tricyclazole",
                "Ensure proper field drainage",
                "Avoid overcrowding plants"
            ],
            "BrownSpot": [
                "Apply balanced fertilizers",
                "Use fungicides like Mancozeb",
                "Improve soil nutrient levels"
            ],
        }

        for tip in treatments.get(output["prediction"], []):
            st.write("✔️", tip)
