import streamlit as st
import cv2
import numpy as np
from model import predict_disease_with_gradcam  # or predict_disease_from_image
from model import is_paddy_leaf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Paddy Disease Finder",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🌾 Paddy Disease Finder")
st.subheader("Early detection & treatment suggestions for paddy crops")

st.markdown("---")

# ---------------- TREATMENT SUGGESTIONS ----------------
TREATMENTS = {
    "BacterialBlight": [
        "Use disease-free seeds",
        "Avoid excessive nitrogen fertilizer",
        "Spray copper-based bactericides",
        "Ensure proper field drainage"
    ],
    "Blast": [
        "Apply recommended fungicides (e.g., Tricyclazole)",
        "Maintain proper spacing between plants",
        "Avoid excessive nitrogen usage",
        "Grow blast-resistant varieties"
    ],
    "BrownSpot": [
        "Apply balanced fertilizers",
        "Use fungicides like Mancozeb",
        "Improve soil nutrient levels",
        "Avoid drought stress"
    ]

}

# ---------------- IMAGE INPUT ----------------
option = st.radio(
    "Choose image input method:",
    ("Upload Image", "Take a Picture")
)

img = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a paddy leaf image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "Take a Picture":
    camera_image = st.camera_input("Take a picture of the paddy leaf")
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ---------------- DISPLAY IMAGE ----------------
if img is not None:
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Selected Paddy Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease"):
        
        if img is None:
            st.error("❌ No image found. Please upload or capture an image.")
            st.stop()

        st.write("🧪 Running leaf validation...")

    # --- BASIC LEAF CHECK ---
        if not is_paddy_leaf(img):
            st.warning(
                "⚠️ This image is not a paddy leaf.\n"
                "Please upload a valid rice leaf image."
            )
            st.stop()

    # --- MODEL PREDICTION ---
        with st.spinner("Analyzing the paddy leaf..."):
            output = predict_disease_with_gradcam(img)


        confidence = output.get("confidence", None)

        if confidence is None:
            st.error("❌ Model failed to produce confidence.")
            st.stop()

        if confidence < 0.75:
            st.warning("⚠️ Low confidence prediction. Please upload a clearer paddy leaf image.")
            st.stop()


        # ---------------- RESULT ----------------
        if output["status"] == "confident":
            disease = output["prediction"]
            confidence = output["confidence"]

            st.success(f"🦠 Disease Detected: **{disease}**")

            # --------- SHOW GRAD-CAM (OPTIONAL) ---------
            if "gradcam" in output:
                st.subheader("🔎 Affected Area (Grad-CAM)")
                st.image(
                    cv2.cvtColor(output["gradcam"], cv2.COLOR_BGR2RGB),
                    use_container_width=True

                )

            # --------- TREATMENT SUGGESTIONS ---------
            st.subheader("🌱 Treatment & Improvement Suggestions")
            for tip in TREATMENTS[disease]:
                st.write("✔", tip)

        else:
            st.warning("⚠️ Prediction Uncertain")
            st.write("Please upload a clearer image with proper lighting.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Paddy Disease Finder | ML-based Early Detection System")
