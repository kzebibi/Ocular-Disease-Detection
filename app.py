import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import mlflow
import yaml
from src.features.build_features import build_features

# Load configuration
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)


logged_model = "models:/best_model/Production"
model = mlflow.keras.load_model(logged_model)

# Get class names
class_names = config["data"]["class_names"]
class_labels = {'ACRIMA': 0, 'Glaucoma': 1, 'ODIR-5K': 2, 'ORIGA': 3, 'cataract': 4, 'retina_disease': 5}
def preprocess_image(image):
    try:
        img = image.resize((224, 224))  # Adjust size as per your model's input
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        equalized_features, statistical_features = build_features(img_array)
        equalized_features = tf.expand_dims(equalized_features, 0)
        statistical_features = tf.expand_dims(statistical_features, 0)
        return equalized_features / 255.0, statistical_features  # Normalize the image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None


def predict(image):
    equalized_features, _ = preprocess_image(image)
    if equalized_features is None:
        return None, None
    try:
        predictions = model.predict(equalized_features)  # Pass only one input
        print("Raw predictions:", predictions)
        score = tf.nn.softmax(predictions[0])
        print("Softmax scores:", score)
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None
    
# Streamlit app layout
st.title("Ocular Disease Detection")
st.write("Upload an image of an eye to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("Classifying...")
        label, confidence = predict(image)
        if label and confidence:
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Failed to classify the image.")
    except Exception as e:
        st.error(f"Error loading image: {e}")