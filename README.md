# Ocular Disease Detection

This project aims to detect ocular diseases using deep learning techniques.

## Project Structure

- `src/`: Source code for data processing, model training, and evaluation.
- `configs/`: Configuration files for model parameters.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `data/`: Directory for raw and processed data.
- `models/`: Directory for saved models.

### Training Guide with MLflow

1. **Setup:**
   ```bash
   # Update system and install Python 3.9
   sudo apt update
   sudo apt install -y python3.9 python3.9-venv python3.9-dev

   # Create and activate a virtual environment
   python3.9 -m venv ~/.venv
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
   ```

2. **Start MLflow Server:**
   - For local SQLite:
     ```bash
     mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 5001
     ```
   - For Azure ML backend, ensure you have the correct URI:
     ```bash
     mlflow ui --backend-store-uri azureml://<your-azure-uri>
     ```

4. **Configure Models:**
   - Edit `configs/model_config.yaml` to ensure configurations for all desired models (e.g., DenseNet121, Xception) are set up correctly. Adjust hyperparameters and paths as needed.

5. **Run the Training Script:**
   - Open a new terminal and navigate to your project root directory. Run:
     ```bash
     python src/main.py
     ```
   - This process will train each model specified in your configuration file and log relevant metrics and artifacts to MLflow.

6. **Monitor Training:**
   - Monitor training progress in the terminal, which will include outputs for each epoch and metrics.

7. **View Results in MLflow:**
   - Access the MLflow UI at `http://localhost:5000`. You can explore experiments, run metrics, parameters, and artifacts.

8. **Compare Models:**
   - Utilize the MLflow UI to select multiple runs for comparative analysis of metrics.

9. **Retrieve the Best Model:**
   - The best model based on validation accuracy is saved in the `models/production/` directory.

10. **Further Analysis:**
    - You can programmatically retrieve and analyze run information using the MLflow API.

11. **Shutdown:**
    - Stop the MLflow server by pressing Ctrl+C in its terminal.

### Additional Tips:
- Modify the `model_config.yaml` to experiment with different hyperparameters and rerun the script.
- Leverage `mlflow.keras.autolog()` in your training code to automatically log Keras metrics.
- Consider setting up a remote MLflow tracking server for team collaboration.

Document any specific setup steps or requirements unique to your project for team use or open-sourcing.
