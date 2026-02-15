# 🌾 Paddy Disease Finder  
### An Explainable Deep Learning System for Paddy Leaf Disease Detection

---

## Abstract

Early detection of paddy (rice) leaf diseases is critical for improving crop yield and reducing economic losses in agriculture.  
This project presents an **end-to-end deep learning system** for automated paddy leaf disease detection using **MobileNetV2** and **explainable artificial intelligence (Grad-CAM)**. The system is deployed as a **Streamlit-based web application** and incorporates robust input validation to prevent incorrect predictions on non-paddy images.

---

## 1. Introduction

Paddy crops are vulnerable to various leaf diseases that spread rapidly under unfavorable conditions. Traditional disease identification relies on expert knowledge and manual inspection, which is time-consuming and error-prone.

This project aims to:
- Automate disease detection from paddy leaf images
- Provide explainable predictions
- Ensure safe inference by rejecting invalid inputs
- Offer treatment suggestions to support farmers

---

## 2. Diseases Considered

The system classifies paddy leaf images into the following categories:

- **Bacterial Blight**
- **Blast**
- **Brown Spot**
- **Non Paddy Objects**

Images that do not belong to paddy leaves are automatically rejected.

---

## 3. Methodology

### 3.1 Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Approach**: Transfer Learning
- **Input Size**: 224 × 224 RGB images
- **Framework**: TensorFlow / Keras

### 3.2 Explainability
- **Technique**: Gradient-weighted Class Activation Mapping (Grad-CAM)
- **Purpose**: Visualize regions influencing model predictions

### 3.3 Input Validation
To avoid incorrect predictions on unrelated images:
- Color-based filtering (green dominance)
- Shape-based analysis (elongated leaf structure)
- Confidence-based rejection

Only validated paddy leaf images are passed to the classifier.

---

## 4. System Architecture

1. **User Image**  
   The user uploads or captures a paddy leaf image using the application interface.

2. **Leaf Validation (Color + Shape)**  
   The system verifies whether the input image corresponds to a valid paddy leaf using color dominance and shape analysis.  
   Non-paddy images are rejected at this stage.

3. **MobileNetV2 Classifier**  
   Validated paddy leaf images are passed to the MobileNetV2 deep learning model for disease classification.

4. **Confidence Thresholding**  
   Predictions with confidence below a predefined threshold are rejected to prevent unreliable results.

5. **Grad-CAM Explanation**  
   Grad-CAM heatmaps are generated to visually explain the regions of the leaf influencing the model’s prediction.

6. **Prediction + Treatment Suggestions**  
   The final output includes the predicted disease, confidence score, and recommended treatment or care suggestions.

---

## 5. Results and Observations

- The model accurately classifies paddy leaf diseases under normal lighting conditions.
- Grad-CAM visualizations confirm that the model focuses on disease-affected regions of the leaf.
- Input validation effectively prevents incorrect predictions on unrelated or non-paddy images.
- The system performs reliably on cloud deployment using CPU-only execution.

---

## 6. Limitations

- The system is limited to paddy (rice) leaves only.
- Performance may degrade for:
   - Very low-quality images
   - Extreme lighting variations
- Other crop species are not supported in the current version.

---

## 7. Future Enhancements

- Extend the system to support multiple crop species.
- Convert the application into a mobile app using Flutter or Android.
- Integrate real-time disease monitoring and alerts.
- Deploy as a REST API for large-scale usage.

---

## 8. Use Cases

- Farmers for early disease detection
- Agricultural researchers and students
- Educational demonstrations of explainable AI
- Prototype for smart agriculture systems

---

## 9. Model UI
<img width="1919" height="1022" alt="Screenshot 2026-02-15 201637" src="https://github.com/user-attachments/assets/320bc002-9d32-4c17-9af3-9b6d0158cec6" />
<img width="1919" height="1079" alt="Screenshot 2026-02-15 201647" src="https://github.com/user-attachments/assets/3abd89c7-a710-46ef-ba26-775bc38dde37" />


## 10. Conclusion

This project demonstrates a practical application of deep learning and explainable AI in agriculture. By combining model accuracy, input validation, and user-friendly deployment, the Paddy Disease Finder provides a reliable solution for early disease detection and decision support.
