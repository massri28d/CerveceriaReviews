"""Project pipelines."""
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from cerveceriar.pipelines.data_processing import pipeline as data_processing_pipeline
from cerveceriar.pipelines.modeling import pipeline as modeling_pipeline  # Agrega esta línea

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "dp": data_processing_pipeline.create_pipeline(),
        "modeling": modeling_pipeline.create_pipeline(),  # Agrega esta línea
        "__default__": data_processing_pipeline.create_pipeline(),
    }
