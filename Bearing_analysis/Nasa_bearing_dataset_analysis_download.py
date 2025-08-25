import kaggle
import os

# Create data directory if it doesn't exist
os.makedirs('./data/raw', exist_ok=True)

# Download dataset
kaggle.api.authenticate()
kaggle.api.dataset_download_files('vinayak123tyagi/bearing-dataset',
                                  path='./data/raw',
                                  unzip=True)

print("Dataset downloaded to ./data/raw")