import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def calculate_k_distance(X_scaled: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Calcula las distancias k más cercanas para cada punto en el dataset.
    """
    print(f"Iniciando cálculo de distancias k más cercanas con k={k}.")
    
    # Configurar el modelo de vecinos más cercanos
    print("Configurando el modelo de vecinos más cercanos.")
    neighbors = NearestNeighbors(n_neighbors=k)
    
    # Entrenar el modelo con los datos
    print("Entrenando el modelo de vecinos más cercanos.")
    neighbors.fit(X_scaled)
    
    # Calcular las distancias y extraer las correspondientes al vecino k-ésimo
    print("Calculando distancias a los k vecinos más cercanos.")
    distances, _ = neighbors.kneighbors(X_scaled)
    
    # Ordenar las distancias para graficar posteriormente
    print("Ordenando distancias calculadas.")
    sorted_distances = np.sort(distances[:, -1])
    
    print("Cálculo de distancias k más cercanas completado.")
    return sorted_distances

def plot_k_distance(distances: np.ndarray, k: int):
    """
    Genera un gráfico de distancias k más cercanas.
    """
    print(f"Iniciando generación del gráfico de k-distance (k={k}).")
    
    # Crear el gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.title(f"Gráfico de K-Distance (k={k})")
    plt.xlabel("Puntos")
    plt.ylabel("Distancia a la k-ésima vecindad más cercana")
    plt.grid(True)  # Agregar cuadrícula para mejorar la visualización
    plt.show()
    
    print("Gráfico de k-distance generado exitosamente.")
