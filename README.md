# Ocular Disease Detection

This project aims to detect ocular diseases using deep learning techniques.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the data preprocessing script: `python src/data/make_dataset.py`
4. Train the model: `python src/models/train_model.py`
5. Evaluate the model: `python src/models/evaluate_model.py`

## Project Structure

- `src/`: Source code for data processing, model training, and evaluation.
- `configs/`: Configuration files for model parameters.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `data/`: Directory for raw and processed data.
- `models/`: Directory for saved models.

## License

This project is licensed under the MIT License.




Certainly! I'll provide you with a step-by-step guide on how to train this project for each model and view the results in MLflow. Here's the process:

1. Setup:
```bash
# Update system and install Python 3.9
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# Create a virtual environment (choose one of the following)
# Option 1: Using venv (for general use)
python3.9 -m venv ~/.venv
# Option 2: Using conda (if you prefer conda)
# conda create --prefix ./venv python=3.9

# Activate the virtual environment
source ~/.venv/bin/activate

# Add the activation command to .bashrc for automatic activation
echo "source ~/.venv/bin/activate" >> ~/.bashrc

# Upgrade pip and install requirements
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# Set up Kaggle API
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
python src/data/download_dataset.py

# Run the main script
python src/main.py

# Start MLflow server (choose one of the following)
# Option 1: Local SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts

# Option 2: Azure ML backend (make sure you have the correct URI)
mlflow ui --backend-store-uri azureml://eastus.api.azureml.ms/mlflow/v2.0/subscriptions/0dce0142-38af-4e0b-a5a1-6ef7b0ed6ac2/resourceGroups/project/providers/Microsoft.MachineLearningServices/workspaces/depiproject
```
   - Ensure you have all the required dependencies installed. You might want to create a `requirements.txt` file with all the necessary packages.
   - Upload the kaggle.json file 
   ```bash
      mkdir ~/.kaggle
      mv /workspaces/Ocular-Disease-Detection/kaggle.json ~/.kaggle/
   ```

2. Prepare the data:
   - Run the data download and preprocessing scripts if you haven't already:
     ```
     python src/data/download_dataset.py
     ```

3. Configure the models:
   - Open `configs/model_config.yaml` and ensure it contains configurations for all the models you want to train (DenseNet121, Xception, and your custom model).
   - Adjust hyperparameters, paths, and other settings as needed.

4. Start MLflow server:
   - Open a terminal and start the MLflow tracking server:
     ```
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
     ```
   - This will start the MLflow UI, typically accessible at `http://localhost:5000`.

5. Run the training script:
   - Open a new terminal window and navigate to your project root directory.
   - Run the main script:
     ```
     python src/main.py
     ```
   - This script will:
     - Train each model specified in the config file.
     - Log metrics, parameters, and artifacts to MLflow for each model.
     - Save the best model based on validation accuracy.

6. Monitor training:
   - While the script is running, you can monitor the progress in the terminal.
   - You'll see output for each epoch, including training and validation metrics.

7. View results in MLflow:
   - Open a web browser and go to `http://localhost:5000`.
   - In the MLflow UI, you'll see an experiment named "ocular_disease_detection" (or whatever you specified in the config).
   - Click on the experiment to see all the runs (one for each model).
   - For each run, you can:
     - View metrics like accuracy, loss, AUC, etc.
     - See parameters used for training.
     - Access artifacts like model files, confusion matrices, and plots.

8. Compare models:
   - In the MLflow UI, you can select multiple runs to compare their metrics side by side.
   - Use the "Compare" feature to create charts comparing different metrics across models.

9. Retrieve the best model:
   - The script automatically selects the best model based on validation accuracy and saves it to the path specified in the config.
   - You can find this model in the `models/production/` directory.

10. Further analysis:
    - You can use the MLflow API to programmatically retrieve run information for further analysis if needed.

11. Shut down:
    - Once you're done, you can stop the MLflow server by pressing Ctrl+C in its terminal window.

Additional tips:
- If you want to run experiments with different hyperparameters, you can modify the `model_config.yaml` file and run the script multiple times.
- You can use MLflow's `mlflow.keras.autolog()` function (which is likely already implemented in your `train_model.py`) to automatically log many Keras metrics without explicit logging statements.
- If you're working in a team, you might want to set up a remote tracking server for MLflow so that everyone can access the results.

Remember to document any specific setup steps or requirements for your project, especially if you're working with a team or plan to open-source the project.