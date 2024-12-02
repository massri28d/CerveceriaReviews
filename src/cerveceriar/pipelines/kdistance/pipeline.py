from kedro.pipeline import Pipeline, node
from .nodes import calculate_k_distance, plot_k_distance

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=calculate_k_distance,
                inputs=["X_scaled", "params:k"],
                outputs="k_distances",
                name="calculate_k_distance",
            ),
            node(
                func=plot_k_distance,
                inputs=["k_distances", "params:k"],
                outputs=None,
                name="plot_k_distance",
            ),
        ]
    )
