import os
import sys
import mlflow
from sklearn.metrics import confusion_matrix
from azureml.core import Workspace, Run
from azureml.mlflow import get_mlflow_tracking_uri

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import create_data_generators
from models.train_model import train_model
from models.evaluate_model import evaluate_model
from utils.config import load_config
from visualization.visualize import (
    plot_training_history,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
)


def update_model_input_shapes(config):
    for model_name in config["models"].keys():
        config["models"][model_name]["input_shape"] = (*config["data"]["image_size"], 4)


def main():
    # Set up Azure ML workspace
    ws = Workspace.from_config()
    
    # Get the current run context
    run = Run.get_context()

    config = load_config("configs/model_config.yaml")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(get_mlflow_tracking_uri(ws))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Add these debug print statements here
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment name: {config['mlflow']['experiment_name']}")
    print(f"Active run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'No active run'}")

    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(
        config["data"]["train_path"],
        config["data"]["test_path"],
        image_size=tuple(config["data"]["image_size"]),
        batch_size=config["data"]["batch_size"],
    )

    # Update the input shape in the config to match your data
    update_model_input_shapes(config)

    class_names = list(train_generator.class_indices.keys())

    best_accuracy = 0
    best_model_name = None

    for model_name in config["models"].keys():
        with mlflow.start_run(run_name=model_name):
            # Train model
            history, trained_model, test_predictions = train_model(model_name, config)

            # Plot and log training history
            plot_training_history(history)
            mlflow.log_artifact("training_history.png")

            # Evaluate model and log results
            y_true, y_pred, y_pred_proba = evaluate_model(
                trained_model, test_generator, class_names
            )

            # Log evaluation metrics
            accuracy = mlflow.get_run(mlflow.active_run().info.run_id).data.metrics[
                "accuracy"
            ]
            mlflow.log_metric("test_accuracy", accuracy)
            run.log(f"{model_name}_test_accuracy", accuracy)

            # Plot and log ROC curve
            plot_roc_curve(y_true, y_pred_proba, class_names)
            mlflow.log_artifact("roc_curve.png")

            # Plot and log Precision-Recall curve
            plot_precision_recall_curve(
                y_true, y_pred_proba, class_names
            )
            mlflow.log_artifact("precision_recall_curve.png")

            # Plot and log confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, class_names)
            mlflow.log_artifact("confusion_matrix.png")

            # Log the model
            mlflow.keras.log_model(trained_model, "model")

            # Update best model if necessary
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name

    # Log the best model
    run.log("best_model", best_model_name)
    run.log("best_accuracy", best_accuracy)

    print(f"Best model: {best_model_name} with accuracy: {best_accuracy}")

    # Load the best model for production
    best_runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string=f"tags.mlflow.runName = '{best_model_name}'",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if best_runs.empty:
        print(f"No runs found for model: {best_model_name}")
        return

    best_run = best_runs.iloc[0]

    # Load the best model for production
    print(
        f"Best model: {best_run.data.tags['mlflow.runName']} with accuracy: {best_run.data.metrics['accuracy']}"
    )

    # Load the best model for production
    try:
        best_model = mlflow.keras.load_model(f"runs:/{best_run.run_id}/model")
        best_model.save(config["paths"]["production_model_path"])

        # Log the production model path
        run.log("production_model_path", config["paths"]["production_model_path"])
    except Exception as e:
        print(f"Error loading or saving the best model: {str(e)}")


if __name__ == "__main__":
    main()
