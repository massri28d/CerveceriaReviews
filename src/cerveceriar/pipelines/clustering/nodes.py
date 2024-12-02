import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import umap
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

# Nodo: Escalado de datos con imputación
def scale_data(data: pd.DataFrame) -> np.ndarray:
    print("Iniciando el proceso de escalado e imputación de datos...")
    data_numeric = data.select_dtypes(include=[np.number])
    if data_numeric.empty:
        raise ValueError("No se encontraron columnas numéricas en los datos de entrada.")
    
    print("Realizando imputación de valores faltantes...")
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_numeric)
    
    print("Aplicando escalado estándar a los datos...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_imputed)
    print("Escalado e imputación completados.")
    return scaled_data

# Nodo: Reducción de dimensionalidad con PCA
def apply_pca(data: np.ndarray, n_components: int = 2, sample_size: int = 10000) -> pd.DataFrame:
    print(f"Iniciando PCA con {n_components} componentes principales...")
    data = data[:sample_size]  # Submuestreo
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print("PCA completado.")
    return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

# Nodo: Reducción de dimensionalidad con UMAP
def apply_umap(data: np.ndarray, n_components: int = 2, sample_size: int = 10000) -> pd.DataFrame:
    print(f"Iniciando UMAP con {n_components} componentes...")
    data = data[:sample_size]  # Submuestreo
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    reduced_data = umap_model.fit_transform(data)
    print("UMAP completado.")
    return pd.DataFrame(reduced_data, columns=[f'UMAP{i+1}' for i in range(n_components)])

# Nodo: DBSCAN con tamaño de muestra opcional
def apply_dbscan(data: np.ndarray, eps: float = 0.7, min_samples: int = 10) -> np.ndarray:
    print(f"Iniciando DBSCAN con eps={eps} y min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    print(f"DBSCAN completado. Se encontraron {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
    return labels

# Nodo: Clustering jerárquico
def hierarchical_clustering(data: np.ndarray, threshold: float = 40):
    print("Iniciando clustering jerárquico...")
    Z = linkage(data, method='ward')
    linkage_matrix_df = pd.DataFrame(Z)
    hierarchical_clusters = fcluster(Z, threshold, criterion='distance')
    print("Clustering jerárquico completado.")
    clusters_df = pd.DataFrame(hierarchical_clusters, columns=["Cluster"])
    return linkage_matrix_df, clusters_df

# Nodo: Visualización de resultados
def plot_clusters(data, clusters, title):
    print(f"Generando gráfico para clusters: {title}...")
    cluster_labels = clusters.flatten() if isinstance(clusters, np.ndarray) else clusters.to_numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap='viridis', s=30)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()
    print(f"Gráfico generado para {title}.")

# Nodo: Codificación de columnas categóricas
def encode_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    print("Iniciando codificación de columnas categóricas...")
    data_copy = data.copy()  # Evitar modificar el DataFrame original
    label_encoder = LabelEncoder()
    for column in data_copy.select_dtypes(include=['object']).columns:
        print(f"Codificando columna categórica: {column}")
        data_copy[column] = label_encoder.fit_transform(data_copy[column])
    print("Codificación de columnas categóricas completada.")
    return data_copy
