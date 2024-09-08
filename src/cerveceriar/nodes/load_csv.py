# src/cerveceriar/nodes/load_csv.py
import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    """Función para cargar un archivo CSV y devolver un DataFrame."""
    df = pd.read_csv(filepath, sep=",")
    return df
