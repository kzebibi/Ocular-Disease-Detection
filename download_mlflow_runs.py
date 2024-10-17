import os
import mlflow
from azureml.core import Workspace
from azureml.mlflow import get_mlflow_tracking_uri

def download_mlflow_runs(experiment_name, local_dir):
    # Set up the Azure ML workspace
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(get_mlflow_tracking_uri(ws))

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Get all runs for the experiment
    runs = mlflow.search_runs()

    for run_id in runs['run_id']:
        run = mlflow.get_run(run_id)
        
        # Create a directory for each run
        run_dir = os.path.join(local_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Download metrics
        with open(os.path.join(run_dir, 'metrics.txt'), 'w') as f:
            for key, value in run.data.metrics.items():
                f.write(f"{key}: {value}\n")

        # Download parameters
        with open(os.path.join(run_dir, 'params.txt'), 'w') as f:
            for key, value in run.data.params.items():
                f.write(f"{key}: {value}\n")

        # Download artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            artifact_path = client.download_artifacts(run_id, artifact.path, run_dir)
            print(f"Downloaded {artifact.path} to {artifact_path}")

    print(f"All runs downloaded to {local_dir}")

# Usage
experiment_name = "ocular_disease_detection"
local_directory = "./mlflow_runs"
download_mlflow_runs(experiment_name, local_directory)