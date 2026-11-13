from kedro.framework.project import configure_project
from purchase_predict.pipeline_registry import register_pipelines


def test_pipeline_registry():
    """Verify that all pipeline modules register cleanly without errors."""
    configure_project("purchase_predict")
    pipelines = register_pipelines()
    assert "__default__" in pipelines
    assert "loading" in pipelines
    assert "processing" in pipelines
    assert "training" in pipelines
