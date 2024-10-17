from azureml.core import Workspace
from azureml.mlflow import get_mlflow_tracking_uri

ws = Workspace.from_config()
tracking_uri = get_mlflow_tracking_uri(ws)
print(tracking_uri)