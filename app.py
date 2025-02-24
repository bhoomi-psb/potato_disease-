import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive file ID (Extracted from your link)
file_id = "1C_HDmxRRh3FC08lJDIyp1MS78xEyPP4m"

# File path to save model
model_path = "trained_plant_disease_model.keras"

# Download the model if not already present
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define class labels for potato leaf diseases
class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']

# Custom CSS for UI improvements
st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(to bottom, #f4e1c6, #d2b48c, #a67b5b);
            color: black;
        }

        /* Title Styling */
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #8B5A2B;
            margin-bottom: 20px;
        }

        /* Upload bar */
        .upload-bar {
            background: #8B5A2B;
            color: white;
            font-size: 18px;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }

        /* Prediction result box */
        .prediction-box {
            font-size: 20px;
            font-weight: bold;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }

        .healthy { background: #d4edda; color: #155724; }
        .warning { background: #fff3cd; color: #856404; }
        .danger  { background: #f8d7da; color: #721c24; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title (now bigger)
st.markdown('<div class="title">🥔 Potato Leaf Disease Classification</div>', unsafe_allow_html=True)

# Upload Instruction inside the brown bar
st.markdown('<div class="upload-bar">Upload an image of a potato leaf to classify its disease.</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=False, width=350)

    # Ensure image is in RGB mode
    image = image.convert("RGB")

    # Preprocess the image
    image = image.resize((128, 128))  # Resize to match model input size
    image_array = np.array(image)  # Keep raw pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    confidence = np.max(predictions)  # Get confidence score

    # Define result styles and messages
    if class_labels[predicted_class] == 'Potato___Early_blight':
        result_class = "warning"
        message = "⚠️ This leaf has Early Blight. Consider using fungicides and improving field management."
    elif class_labels[predicted_class] == 'Potato___Late_blight':
        result_class = "danger"
        message = "🚨 This leaf has Late Blight. Immediate action is needed to prevent crop loss!"
    else:
        result_class = "healthy"
        message = "✅ This potato leaf is healthy!"

    # Display prediction results with styled box
    st.markdown(f"""
        <div class="prediction-box {result_class}">
            <p><strong>Predicted Class:</strong> {class_labels[predicted_class]}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}</p>
            <p>{message}</p>
        </div>
    """, unsafe_allow_html=True)
