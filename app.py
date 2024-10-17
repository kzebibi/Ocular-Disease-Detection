import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import mlflow
from mlflow.tracking import MlflowClient
import yaml
from src.features.build_features import build_features

# Load configuration
with open("configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set up MLflow client
client = MlflowClient()
experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])

# Find the best model
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"]
)
best_run = runs[0]

# Load the best model
model = mlflow.keras.load_model(f"runs:/{best_run.info.run_id}/model")

# Get class names
class_names = config["data"]["class_names"]


def preprocess_image(image):
    img = image.resize((224, 224))  # Adjust size as per your model's input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    equalized_features, statistical_features = build_features(img_array)
    equalized_features = tf.expand_dims(equalized_features, 0)
    statistical_features = tf.expand_dims(statistical_features, 0)
    return equalized_features / 255.0, statistical_features  # Normalize the image


def predict(image):
    equalized_features, statistical_features = preprocess_image(image)
    predictions = model.predict([equalized_features, statistical_features])
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence


st.title("Ocular Disease Detection")

st.write("Upload an image of an eye to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, confidence = predict(image)
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")

st.write("---")
st.write("Model Information:")
st.write(f"Best model run name: {best_run.data.tags['mlflow.runName']}")
st.write(f"Best model accuracy: {best_run.data.metrics['accuracy']:.4f}")
