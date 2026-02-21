import grafial
from _helpers import load_example


def test_minimal_flow_exports_graph(tmp_path):
    source = load_example("crates/grafial-examples/minimal.grafial")
    program = grafial.compile(source)
    ctx = grafial.run_flow(program, "MinimalFlow")

    # Should export a graph named "output"
    g = ctx.get_graph("output")
    assert g is not None

    # Basic iteration works
    nodes = list(g.nodes())
    edges = list(g.edges())
    assert len(nodes) >= 1
    assert len(edges) >= 1

    # NodeView methods
    n0 = nodes[0]
    assert isinstance(n0.id, str)
    assert isinstance(n0.label, str)
    assert isinstance(n0.has_attr("value"), bool)
    if n0.has_attr("value"):
        assert isinstance(n0.E("value"), float)
        assert isinstance(n0.Var("value"), float)


def test_minimal_pandas_and_networkx_exports_if_installed():
    source = load_example("crates/grafial-examples/minimal.grafial")
    program = grafial.compile(source)
    ctx = grafial.run_flow(program, "MinimalFlow")
    g = ctx.get_graph("output")

    # pandas optional
    try:
        import pandas as pd  # noqa: F401
    except Exception:  # pragma: no cover - optional dep
        pass
    else:
        nodes_df, edges_df = g.to_pandas()
        assert hasattr(nodes_df, "__class__")
        assert hasattr(edges_df, "__class__")
        assert len(nodes_df) >= 1
        assert len(edges_df) >= 1

    # networkx optional
    try:
        import networkx as nx  # noqa: F401
    except Exception:  # pragma: no cover - optional dep
        pass
    else:
        G = g.to_networkx(threshold=0.0)
        assert hasattr(G, "nodes")
        assert G.number_of_nodes() >= 1
        assert G.number_of_edges() >= 1
