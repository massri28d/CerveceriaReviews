from kedro.pipeline import Pipeline, node
from .nodes import (
    normalize_data,
    impute_missing_values,
    calculate_inertia,
    plot_elbow,
    calculate_silhouette,
    plot_silhouette,
    generate_labels,  # Nueva función para generar etiquetas
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalize_data,
                inputs="filtered_review_overall",
                outputs="X_scaled",
                name="normalize_data",
            ),
            node(impute_missing_values, "X_scaled", "X_imputed"),
            node(generate_labels, ["X_imputed", "params:k_values"], "labels"),
            node(
                func=calculate_inertia,
                inputs=["X_imputed", "params:k_values"],
                outputs="inertia",
                name="calculate_inertia",
            ),
            node(
                func=plot_elbow,
                inputs=dict(k_values="params:k_values", inertia="inertia"),
                outputs=None,
                name="plot_elbow",
            ),
             node(
                func=calculate_silhouette,
                inputs=dict(X="X_imputed", labels="labels"),  # Cambia X_scaled por X_imputed
                outputs="silhouette_scores",
                name="calculate_silhouette",
            ),
            node(
                func=plot_silhouette,
                inputs=dict(k_values="params:k_values", silhouette_scores="silhouette_scores"),
                outputs=None,
                name="plot_silhouette",
            ),
        ]
    )
