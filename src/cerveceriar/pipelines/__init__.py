from .modeling import pipeline as modeling_pipeline

def register_pipelines():
    return {
        "__default__": modeling_pipeline.create_pipeline(),
        "modeling": modeling_pipeline.create_pipeline(),
    }
