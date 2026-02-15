import streamlit as st
from PIL import Image
from model import load_trained_model, predict_image, DISEASE_INFO

# Page Configuration
st.set_page_config(page_title="Paddy Health Guard", page_icon="🌾")

st.title("🌾 Paddy Leaf Disease Detector")
st.markdown("Identify paddy diseases instantly using AI.")

# Load Model with caching to prevent slow reloads
@st.cache_resource
def get_model():
    return load_trained_model("rice_model.keras")

try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'rice_model.keras' is in the same folder.")
    st.stop()

# Input Selection
option = st.selectbox("How would you like to provide the image?", 
                      ("Upload from Gallery", "Use Camera"))

image = None

if option == "Upload from Gallery":
    # FIXED: Changed from file_file_uploader to file_uploader
    uploaded_file = st.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Snap a photo of the leaf")
    if camera_file:
        image = Image.open(camera_file)

# Prediction Result
if image:
    st.image(image, caption="Captured Image", use_container_width=True)
    
    if st.button("Analyze Leaf"):
        with st.spinner('AI is analyzing...'):
            label, confidence = predict_image(model, image)
        
        # Display Logic
        if label == "Non_Paddy":
            st.error(f"Result: {label.replace('_', ' ')}")
            st.warning("⚠️ This does not appear to be a paddy leaf. Our AI only analyzes paddy plants. Please try a different photo.")
        else:
            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence Level: **{confidence:.2%}**")
            
            # Show Treatment Details
            info = DISEASE_INFO.get(label)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("💊 Treatment")
                st.write(info["treatment"])
            with col2:
                st.subheader("🛡️ Prevention")
                st.write(info["prevention"])

