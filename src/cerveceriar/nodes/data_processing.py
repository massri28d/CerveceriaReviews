
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict

def process_beer_reviews(beer_reviews: pd.DataFrame) -> pd.DataFrame:
    # Asigna el dataset beer_reviews a la variable df
    df = beer_reviews

    # Muestra las primeras filas del dataset para verificar que se haya cargado correctamente
    print(df.head())

    # Aquí puedes aplicar cualquier lógica adicional de procesamiento que necesites
    # Por ejemplo, eliminar filas nulas o seleccionar ciertas columnas
    df_cleaned = df.dropna()

    return df_cleaned


# Función para análisis inicial
def analyze_data(beer_reviews: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Función para análisis inicial del dataset, imprimiendo detalles clave como
    tamaño, columnas y estadísticas.
    """
    print(f"Tamaño del dataset: {beer_reviews.shape}")
    print(f"Columnas del dataset: {beer_reviews.columns}")
    print(f"Información del dataset:")
    print(beer_reviews.info())
    print(f"Descripción estadística:")
    print(beer_reviews.describe())

    # Aquí podríamos retornar información clave si se necesita usar en otros nodos
    return {
        "beer_reviews": beer_reviews
    }

# Función para mostrar gráficos KDE de varias columnas
def plot_kde(beer_reviews: pd.DataFrame):
    """
    Crea gráficos KDE para las columnas de interés en el dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(beer_reviews['review_overall'], label='Review Overall')
    sns.kdeplot(beer_reviews['review_taste'], label='Review Taste')
    sns.kdeplot(beer_reviews['review_appearance'], label='Review Appearance')
    sns.kdeplot(beer_reviews['review_palate'], label='Review Palate')
    plt.title("Distribuciones KDE")
    plt.legend()
    plt.show()

# Función para calcular estadísticas
def calculate_statistics(beer_reviews: pd.DataFrame):
    """
    Calcula media, mediana y moda de diferentes características en el dataset.
    """
    stats_data = {}
    
    for column in ['review_overall', 'review_taste', 'review_appearance', 'review_palate']:
        media = beer_reviews[column].mean()
        mediana = beer_reviews[column].median()
        moda = beer_reviews[column].mode()[0]

        print(f'\n--- {column} ---')
        print(f'Media: {media}')
        print(f'Mediana: {mediana}')
        print(f'Moda: {moda}')
        print(f'Estadísticas descriptivas:\n{beer_reviews[column].describe()}')

        stats_data[column] = {
            'mean': media,
            'median': mediana,
            'mode': moda
        }

    return stats_data

# Función para visualizar matriz de correlación
def plot_correlation_matrix(beer_reviews: pd.DataFrame):
    """
    Crea una matriz de correlación para columnas numéricas en el dataset.
    """
    numeric_columns = ['brewery_id', 'review_time', 'review_overall', 'review_aroma', 
                       'review_appearance', 'review_palate', 'review_taste', 'beer_abv']
    
    correlation_matrix = beer_reviews[numeric_columns].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlación entre Variables Numéricas')
    plt.show()
    
# Función para filtrar datos
def filter_reviews(beer_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra las reseñas con calificaciones mayores o iguales a 4.5.
    """
    filtered_df = beer_reviews[beer_reviews['review_overall'] >= 4.5]
    print(f"Reseñas filtradas (>= 4.5): {filtered_df.shape}")
    
    return filtered_df

