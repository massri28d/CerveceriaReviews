# src/<project_name>/nodes/kaggle_download.py

from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

def download_from_kaggle(dataset: str, destination: str) -> None:
    api = KaggleApi()
    api.authenticate()
    
    # Descarga el dataset
    api.dataset_download_files(dataset, path=destination, unzip=True)
    
    # Si el archivo descargado es un ZIP, descomprímelo
    zip_file_path = os.path.join(destination, f'{dataset}.zip')
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        os.remove(zip_file_path)  # Opcional: eliminar el archivo ZIP después de descomprimir
