"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from cerveceriar.pipelines.data_processing import pipeline as data_processing_pipeline

def register_pipelines():
    return {
        "dp": data_processing_pipeline.create_pipeline(),
        "__default__": data_processing_pipeline.create_pipeline(),
    }
