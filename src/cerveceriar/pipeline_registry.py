"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from cerveceriar.pipelines.data_processing import pipeline as data_processing_pipeline
from cerveceriar.pipelines.modeling import pipeline as modeling_pipeline
from cerveceriar.pipelines.kmeans.pipeline import create_pipeline as create_kmeans_pipeline
from cerveceriar.pipelines.kdistance.pipeline import create_pipeline as create_kdistance_pipeline
from cerveceriar.pipelines.clustering.pipeline import create_pipeline as create_clustering_pipeline
from cerveceriar.pipelines.data_science.pipeline import create_pipeline as create_tada_science_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    # Registro de pipelines
    pipelines = {
        "dp": data_processing_pipeline.create_pipeline(),
        "modeling": modeling_pipeline.create_pipeline(),
        "kmeans": create_kmeans_pipeline(),
        "kdistance": create_kdistance_pipeline(),
        "clustering": create_clustering_pipeline(),
        "data_science": create_tada_science_pipeline(),
    }

    # Agregar un pipeline por defecto
    pipelines["__default__"] = (
        pipelines["dp"]
        + pipelines["modeling"]
        + pipelines["kmeans"]
        + pipelines["kdistance"]
    )

    return pipelines
