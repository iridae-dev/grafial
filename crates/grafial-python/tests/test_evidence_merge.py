import grafial
from _helpers import load_example


def test_runtime_evidence_observations_affect_graph():
    source = load_example("crates/grafial-examples/minimal.grafial")
    prog = grafial.compile(source)

    # Add a new observation at runtime for a new node
    ev = grafial.Evidence("Runtime", model="MinimalBeliefs")
    ev.observe_numeric("Entity", "C", "value", 2.0)

    ctx = grafial.run_flow_with_evidence(prog, "MinimalFlow", ev)
    g = ctx.get_graph("output")
    assert g is not None

    # At least one node should have E[value] near 2.0 (weak prior + observation)
    e_values = [n.E("value") for n in g.nodes() if n.has_attr("value")]
    assert any(abs(v - 2.0) < 0.2 for v in e_values)
