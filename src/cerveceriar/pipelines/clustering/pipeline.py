from kedro.pipeline import Pipeline, node
from .nodes import (
    scale_data,
    apply_pca,
    apply_umap,
    apply_dbscan,
    hierarchical_clustering,
    plot_clusters,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Cambiar "raw_data" por "filtered_review_overall"
            node(func=scale_data, inputs="filtered_review_overall", outputs="scaled_data", name="scale_data_node"),
            node(func=apply_pca, inputs="scaled_data", outputs="pca_data", name="pca_node"),
            node(func=apply_umap, inputs="scaled_data", outputs="umap_data", name="umap_node"),
            node(func=apply_dbscan, inputs="pca_data", outputs="dbscan_clusters", name="dbscan_node"),
            node(
                func=hierarchical_clustering,
                inputs="pca_data",
                outputs=["linkage_matrix", "hierarchical_clusters"],
                name="hierarchical_clustering_node",
            ),
            node(
                func=plot_clusters,
                inputs=dict(data="pca_data", clusters="dbscan_clusters", title="params:plot_titles.dbscan"),
                outputs=None,
                name="plot_dbscan_clusters_node",
            ),
            node(
                func=plot_clusters,
                inputs=dict(data="pca_data", clusters="hierarchical_clusters", title="params:plot_titles.hierarchical"),
                outputs=None,
                name="plot_hierarchical_clusters_node",
            ),
        ]
    )

