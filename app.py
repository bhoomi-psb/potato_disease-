import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID and model path
file_id = "1cJqVRrS5yXXVDE87ZY1jgdimUf85qeWl"
model_path = "trained_plant_disease_model.keras"

# ✅ Download the model if it doesn't exist
if not os.path.exists(model_path):
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Model download failed: {e}")

# ✅ Cached function to load the model (prevents reloading in Streamlit)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load the model
model = load_model()

# Class labels
class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']

# ✅ Custom CSS for Styling
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(to bottom, #f4e1c6, #d2b48c, #a67b5b); color: black; }
        .title { font-size: 36px; font-weight: bold; text-align: center; color: #8B5A2B; margin-bottom: 20px; }
        .upload-bar { background: #8B5A2B; color: white; font-size: 18px; padding: 12px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
        .prediction-box { font-size: 20px; font-weight: bold; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .healthy { background: #d4edda; color: #155724; }
        .warning { background: #fff3cd; color: #856404; }
        .danger  { background: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Title
st.markdown('<div class="title">🥔 Potato Leaf Disease Classification</div>', unsafe_allow_html=True)

# ✅ Upload Instruction Bar
st.markdown('<div class="upload-bar">Upload an image of a potato leaf to classify its disease.</div>', unsafe_allow_html=True)

# ✅ File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if model is None:
        st.error("❌ Model failed to load. Please restart the app or check the model file.")
    else:
        # ✅ Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300, use_container_width=False)

        # Convert image to RGB and resize
        image = image.convert("RGB").resize((128, 128))
        image_array = np.expand_dims(np.array(image), axis=0)

        # ✅ Make Prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # ✅ Prediction Result Display
        if class_labels[predicted_class] == 'Potato___Early_blight':
            result_class = "warning"
            message = "⚠️ This leaf has Early Blight. Consider using fungicides and improving field management."
        elif class_labels[predicted_class] == 'Potato___Late_blight':
            result_class = "danger"
            message = "🚨 This leaf has Late Blight. Immediate action is needed to prevent crop loss!"
        else:
            result_class = "healthy"
            message = "✅ This potato leaf is healthy!"

        # ✅ Display Prediction Box
        st.markdown(f"""
            <div class="prediction-box {result_class}">
                <p><strong>Predicted Class:</strong> {class_labels[predicted_class]}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                <p>{message}</p>
            </div>
        """, unsafe_allow_html=True)
