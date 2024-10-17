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

if experiment is None:
    st.error("Experiment not found. Please check the experiment name in the configuration.")
else:
    # Find the best model
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"]
    )

    if not runs:
        st.error("No runs found in the experiment. Please ensure that the experiment has completed runs.")
    else:
        best_run = runs[0]

        # Load the best model
        model = mlflow.keras.load_model(f"runs:/{best_run.info.run_id}/model")

        # Get class names
        class_names = config["data"]["class_names"]

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
            equalized_features, statistical_features = preprocess_image(image)
            if equalized_features is None or statistical_features is None:
                return None, None
            try:
                predictions = model.predict([equalized_features, statistical_features])
                score = tf.nn.softmax(predictions[0])
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

        st.write("---")
        st.subheader("Model Information")
        st.write(f"**Best model run name**: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
        st.write(f"**Best model accuracy**: {best_run.data.metrics.get('accuracy', 'N/A'):.4f}")
        st.write(f"**Model ID**: {best_run.info.run_id}")
        st.write(f"**Experiment ID**: {experiment.experiment_id}")
        st.write(f"**Model Path**: runs:/{best_run.info.run_id}/model")

        st.write("---")
        st.write("For more information, visit the [MLflow experiment page](https://mlflow.org/).")