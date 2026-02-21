import pytest

import grafial

from _helpers import load_example


def test_compile_error_mapping_parse_and_validation():
    with pytest.raises(ValueError, match="Parse error"):
        grafial.compile("this is not valid grafial syntax")

    invalid_model_target = """
    schema S { node N { } edge E { } }
    belief_model B on MissingSchema {
      edge E { exist ~ Bernoulli(prior=0.5, weight=2.0) }
    }
    """
    with pytest.raises(ValueError, match="Validation error"):
        grafial.compile(invalid_model_target)


def test_program_context_getters_and_runtime_error_mapping():
    source = load_example("crates/grafial-examples/minimal.grafial")
    program = grafial.compile(source)
    assert "Program(" in repr(program)

    with pytest.raises(RuntimeError, match="unknown flow"):
        grafial.run_flow(program, "MissingFlow")

    ctx = grafial.run_flow(program, "MinimalFlow")
    assert "Context(" in repr(ctx)

    graphs = ctx.graphs
    metrics = ctx.metrics
    assert isinstance(graphs, dict)
    assert isinstance(metrics, dict)
    assert "output" in graphs

    assert ctx.get_graph("output") is not None
    assert ctx.get_graph("missing") is None
    assert ctx.get_metric("missing") is None
