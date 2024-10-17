import numpy as np
import mlflow
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
)
import tensorflow as tf
import os


def evaluate_model(model, test_generator, class_names):
    # Predict probabilities
    Y_pred_proba = model.predict(test_generator)
    y_pred = np.argmax(Y_pred_proba, axis=1)

    # Get true labels
    if hasattr(test_generator, "classes"):
        y_true = test_generator.classes
    else:
        # If 'classes' attribute is not available, we need to generate true labels
        y_true = []
        for _, y in test_generator:
            y_true.extend(np.argmax(y, axis=1))
        y_true = np.array(y_true)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy score: {accuracy:.4f}")
    mlflow.log_metric("accuracy", accuracy)

    # Calculate ROC AUC score
    auc = roc_auc_score(y_true, Y_pred_proba, multi_class="ovo")
    print(f"AUC: {auc:.4f}")
    mlflow.log_metric("auc", auc)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save confusion matrix to a file, then log it
    np.save("confusion_matrix.npy", cm)
    mlflow.log_artifact("confusion_matrix.npy")
    os.remove("confusion_matrix.npy")  # Clean up the file after logging

    # Classification Report
    cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification Report:")
    print(cr)
    mlflow.log_text(cr, "classification_report.txt")

    log_classification_metrics(y_true, y_pred, class_names)

    # Calculate and log per-class AUC
    for i, class_name in enumerate(class_names):
        class_auc = roc_auc_score((y_true == i).astype(int), Y_pred_proba[:, i])
        mlflow.log_metric(f"{class_name}_auc", class_auc)

    return y_true, y_pred, Y_pred_proba


def log_model_summary(model):
    # Log model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)
    mlflow.log_text(model_summary, "model_summary.txt")

    # Log model architecture diagram
    tf.keras.utils.plot_model(model, to_file="model_architecture.png", show_shapes=True)
    mlflow.log_artifact("model_architecture.png")


def log_sample_predictions(model, test_generator, class_names, num_samples=5):
    # Get a batch of test data
    x, y_true = next(test_generator)
    y_pred = model.predict(x)

    for i in range(min(num_samples, len(x))):
        true_class = class_names[np.argmax(y_true[i])]
        pred_class = class_names[np.argmax(y_pred[i])]
        confidence = np.max(y_pred[i])

        mlflow.log_metric(f"sample_{i+1}_confidence", confidence)
        mlflow.log_param(f"sample_{i+1}_true_class", true_class)
        mlflow.log_param(f"sample_{i+1}_pred_class", pred_class)


def evaluate_and_log_model(model, test_generator, class_names):
    y_true = []
    y_pred = []
    y_pred_proba = []

    for i in range(len(test_generator)):
        X, y = test_generator[i]
        predictions = model.predict(X)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    return y_true, y_pred, y_pred_proba


def log_classification_metrics(y_true, y_pred, class_names):
    cr_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    for class_name in class_names:
        mlflow.log_metric(f"{class_name}_precision", cr_dict[class_name]["precision"])
        mlflow.log_metric(f"{class_name}_recall", cr_dict[class_name]["recall"])
        mlflow.log_metric(f"{class_name}_f1-score", cr_dict[class_name]["f1-score"])
