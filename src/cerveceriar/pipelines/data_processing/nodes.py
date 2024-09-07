import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Función para procesar el dataset de revisiones de cerveza.
    
    Args:
        data: Un DataFrame de pandas que contiene las revisiones de cerveza.
    
    Returns:
        Un DataFrame de pandas procesado.
    """
    # Ejemplo: Eliminar filas con valores nulos
    processed_data = data.dropna()
    
    return processed_data

def get_top_reviews(data: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """
    Función para obtener los mejores n reviews basados en el review_taste.
    
    Args:
        data: Un DataFrame de pandas que contiene las revisiones de cerveza.
        n: Número de mejores reviews a retornar.
    
    Returns:
        Un DataFrame de pandas con los mejores reviews.
    """
    top_reviews = data.nlargest(n, 'review_taste')[['beer_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_style']]
    return top_reviews

def plot_top_reviews(data: pd.DataFrame):
    """
    Función para visualizar los mejores reviews.
    
    Args:
        data: Un DataFrame de pandas que contiene los mejores reviews.
    """
    # Convertir a formato largo para la visualización
    melted_data = data.melt(id_vars='beer_name', value_vars=['review_aroma', 'review_appearance', 'review_palate', 'review_taste'])

    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_data, x='beer_name', y='value', hue='variable')
    plt.title('Top 50 Beer Reviews by Attributes')
    plt.xlabel('Beer Name')
    plt.ylabel('Review Score')
    plt.xticks(rotation=45)
    plt.legend(title='Review Attribute')
    plt.tight_layout()
    plt.show()







