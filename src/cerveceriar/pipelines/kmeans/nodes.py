import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def preprocess_kmeans_data(filtered_review_overall):
    print("Iniciando preprocesamiento de datos para K-Means.")
    
    # Seleccionar solo las columnas numéricas
    print("Seleccionando columnas numéricas del DataFrame.")
    numeric_data = filtered_review_overall.select_dtypes(include=['number'])

    # Manejar datos faltantes (si aplica)
    print("Llenando valores faltantes con 0.")
    numeric_data = numeric_data.fillna(0)

    # Escalado de los datos numéricos
    print("Escalando datos numéricos con StandardScaler.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # Convertir a DataFrame
    print("Convirtiendo datos escalados a un DataFrame.")
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_data.columns)

    print("Preprocesamiento completado.")
    # Retornar datos limpios y escalados
    return X_scaled_df, numeric_data


def perform_kmeans(X_scaled: np.ndarray, n_clusters: int = 4) -> pd.DataFrame:
    print(f"Iniciando K-Means con {n_clusters} clusters.")
    
    # Ejecutar algoritmo K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)  # Configurar `n_init` para evitar advertencias
    print("Entrenando modelo K-Means...")
    kmeans.fit(X_scaled)
    
    print("K-Means completado. Retornando etiquetas de clusters y centros.")
    return kmeans.labels_, kmeans.cluster_centers_


def reduce_dimensionality(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    print(f"Iniciando reducción de dimensionalidad con PCA ({n_components} componentes).")
    
    # Reducir dimensionalidad
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Reducción de dimensionalidad completada.")
    return X_pca


def create_kmeans_results(data: pd.DataFrame, clusters: np.ndarray, X_pca: np.ndarray) -> pd.DataFrame:
    print("Creando DataFrame con los resultados de K-Means.")
    
    # Crear copias del DataFrame para evitar conflictos de nombres
    print("Copiando DataFrame original.")
    data_result = data.copy()

    # Agregar información de clusters y componentes principales
    print("Agregando etiquetas de clusters y componentes principales (PCA).")
    data_result["Cluster"] = clusters
    data_result["PCA1"] = X_pca[:, 0]
    data_result["PCA2"] = X_pca[:, 1]

    print("Resultados de K-Means preparados.")
    return data_result
