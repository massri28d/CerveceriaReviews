# pipelines/kmeans/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_kmeans_data, perform_kmeans, reduce_dimensionality, create_kmeans_results

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_kmeans_data,
                inputs="filtered_review_overall",
                outputs=["X_scaled", "data_cleaned"],
                name="preprocess_data",
            ),
            node(
                func=perform_kmeans,
                inputs=dict(X_scaled="X_scaled", n_clusters="params:n_clusters"),
                outputs=["clusters", "centers"],
                name="perform_kmeans",
            ),
            node(
                func=reduce_dimensionality,
                inputs="X_scaled",
                outputs="X_pca",
                name="reduce_dimensionality",
            ),
            node(
                func=create_kmeans_results,
                inputs=["data_cleaned", "clusters", "X_pca"],
                outputs="kmeans_results",
                name="create_results",
            ),
        ]
    )
