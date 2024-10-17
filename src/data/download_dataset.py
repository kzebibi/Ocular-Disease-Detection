import os
import kaggle
import zipfile

def download_kaggle_dataset(dataset_name, download_path):
    # Create the download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Download the dataset
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=False)

    print(f"Dataset downloaded to {download_path}")

def extract_zip_files(download_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)
    for file in os.listdir(download_path):
        if file.endswith('.zip'):
            file_path = os.path.join(download_path, file)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            os.remove(file_path)
    
    print(f"All zip files extracted to {extract_path} and removed from {download_path}")

def main():
    # Dataset for different eye diseases
    dataset_name = 'dhirajmwagh1111/dataset-for-different-eye-disease'
    download_path = os.path.join('data', 'raw')
    extract_path = os.path.join('data', 'processed')

    download_kaggle_dataset(dataset_name, download_path)
    extract_zip_files(download_path, extract_path)

if __name__ == "__main__":
    main()
