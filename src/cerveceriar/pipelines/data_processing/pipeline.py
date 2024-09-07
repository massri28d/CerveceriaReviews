from kedro.pipeline import Pipeline, node
from .nodes import load_and_process_data, get_top_reviews, plot_top_reviews



def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_process_data,  # Función para cargar y procesar los datos
            inputs="beer_reviews",        # Nombre del dataset definido en catalog.yml
            outputs="processed_data",    # Nombre del dataset procesado
            name="data_processing_node"  # Nombre opcional del nodo
        ),
        node(
            func=get_top_reviews,
            inputs="processed_data",
            outputs="top_reviews",
            name="top_reviews_node"
        ),
        node(
            func=plot_top_reviews,
            inputs="top_reviews",
            outputs=None,  # No se necesita salida aquí
            name="plot_reviews_node"
        )
    ])
