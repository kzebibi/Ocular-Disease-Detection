import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data.preprocess import create_data_generators
from models.model import create_model, compile_model
import tensorflow as tf
from azureml.core import Run

def create_callbacks(config, model_name):
    return [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-11,
            patience=config["training"]["early_stopping_patience"],
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["training"]["reduce_lr_factor"],
            patience=config["training"]["reduce_lr_patience"],
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=f"{config['paths']['model_save_dir']}/{model_name}_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

def train_model(model_name, config):
    # Get the current run context
    run = Run.get_context()

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name, nested=True):
        train_generator, validation_generator, test_generator = create_data_generators(
            config["data"]["train_path"],
            config["data"]["test_path"],
            image_size=tuple(config["models"][model_name]["input_shape"][:2]),
            batch_size=config["data"]["batch_size"],
        )

        model = create_model(model_name, config)
        model = compile_model(model, model_name, config)

        callbacks = create_callbacks(config, model_name)

        mlflow.keras.autolog()

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=config["training"]["epochs"],
            callbacks=callbacks,
        )

        # Log metrics using both MLflow and AzureML
        for metric_name, metric_value in history.history.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value[-1])
            run.log(f"train_{metric_name}", metric_value[-1])

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_generator)
        
        # Log test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        run.log("test_loss", test_loss)
        run.log("test_accuracy", test_accuracy)

        # Log the best validation accuracy
        best_val_accuracy = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        run.log("best_val_accuracy", best_val_accuracy)

        # Generate predictions on the test set
        test_predictions = model.predict(test_generator)
        
        # You might want to save these predictions or process them further
        # For example, you could save them to a file or compute additional metrics

        return history, model, test_predictions
