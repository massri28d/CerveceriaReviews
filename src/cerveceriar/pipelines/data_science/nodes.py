from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los datos numéricos del DataFrame usando StandardScaler.
    """
    print("Normalizando los datos...")
    numeric_data = data.select_dtypes(include=["number"])  # Seleccionar únicamente columnas numéricas
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    print("Datos normalizados exitosamente.")
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=data.index)
    return scaled_df


def impute_missing_values(X):
    """
    Imputa valores faltantes en los datos utilizando la media de cada columna.
    """
    print("Imputando valores faltantes...")
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    print("Valores faltantes imputados exitosamente.")
    return X_imputed


def calculate_inertia(X_scaled, k_values):
    """
    Calcula la inercia para diferentes valores de k.
    """
    print("Calculando la inercia para diferentes valores de k...")
    inertia = []
    for k in k_values:
        print(f"Entrenando KMeans para k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    print("Cálculo de inercia completado.")
    return inertia


def plot_elbow(k_values, inertia):
    """
    Genera y muestra un gráfico de codo.
    """
    print("Generando gráfico del método del codo...")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(k_values, inertia, marker="o")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inercia")
    plt.title("Método del Codo")
    plt.show()  # Muestra el gráfico
    plt.savefig("elbow_plot.png")  # También guarda el gráfico
    plt.close()
    print("Gráfico del método del codo generado y guardado como 'elbow_plot.png'.")


def generate_labels(X, k_values):
    """
    Genera etiquetas de clusters utilizando KMeans con el mejor valor de k.
    """
    print("Generando etiquetas para los clusters...")
    best_k = k_values[np.argmin(k_values)]  # Por simplicidad, elige el menor k
    print(f"El valor óptimo de k seleccionado es {best_k}.")
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    print("Etiquetas generadas exitosamente.")
    return labels


def calculate_silhouette(X, labels):
    """
    Calcula el puntaje de silueta.
    """
    print("Calculando el puntaje de silueta...")
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(X, labels)
    print(f"Puntaje de silueta calculado: {silhouette:.4f}")
    return silhouette


def plot_silhouette(k_values, silhouette_scores):
    """
    Genera y muestra un gráfico de puntajes de silueta.
    """
    print("Generando gráfico del puntaje de silueta...")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Puntaje de Silueta")
    plt.title("Puntaje de Silueta por Número de Clusters")
    plt.show()  # Muestra el gráfico
    plt.savefig("silhouette_plot.png")  # También guarda el gráfico
    plt.close()
    print("Gráfico del puntaje de silueta generado y guardado como 'silhouette_plot.png'.")

