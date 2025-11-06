import grafial
from pathlib import Path


def load_example(path: str) -> str:
    # Find project root: tests/ is in crates/grafial-python/, so go up 2 levels
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent
    example_path = project_root / path
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read()


def test_pipeline_stages_and_metric_exports():
    source = load_example("crates/grafial-examples/pipeline_composition.grafial")
    prog = grafial.compile(source)

    # Stage 1
    ctx1 = grafial.run_flow(prog, "CleaningStage")
    assert ctx1.get_graph("cleaned_graph") is not None
    # Metric export should exist
    m = ctx1.get_metric("cleaning_stats")
    assert m is not None
    assert isinstance(m, float)

    # Stage 2 with prior context
    ctx2 = grafial.run_flow_with_context(prog, "EnrichmentStage", ctx1)
    assert ctx2.get_graph("enriched_graph") is not None
    m2 = ctx2.get_metric("enrichment_stats")
    assert m2 is not None
    assert isinstance(m2, float)

    # Stage 3 with prior context
    ctx3 = grafial.run_flow_with_context(prog, "QualityAnalysis", ctx2)
    assert ctx3.get_graph("final_result") is not None

