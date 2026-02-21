import builtins
import sys

import pytest

import grafial


FORCED_EDGE_SOURCE = """
schema TestGraph {
  node Entity {
    value: Real
  }
  edge CONNECTED { }
}

belief_model TestBeliefs on TestGraph {
  node Entity {
    value ~ Gaussian(mean=0.0, precision=1.0)
  }
  edge CONNECTED {
    exist ~ Bernoulli(prior=0.5, weight=2.0)
  }
}

evidence TestEvidence on TestBeliefs {
  Entity {
    "A" { value: 1.0 },
    "B" { value: 2.0 }
  }
  CONNECTED(Entity -> Entity) { "A" -> "B" }
}

rule ForceAbsent on TestBeliefs {
  pattern
    (A:Entity)-[e:CONNECTED]->(B:Entity)
  where
    prob(e) >= 0.0
  action {
    delete e confidence=high
  }
  mode: for_each
}

flow TestFlow on TestBeliefs {
  graph base = from_evidence TestEvidence
  graph g = base |> apply_rule ForceAbsent
  export g as "out"
}
"""


def _block_import(monkeypatch, module_name: str):
    real_import = builtins.__import__

    # Ensure import will consult the import hook instead of sys.modules cache.
    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(f"{module_name}."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name or name.startswith(f"{module_name}."):
            raise ImportError(f"blocked import for {module_name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_node_and_edge_views_cover_exposed_methods():
    program = grafial.compile(FORCED_EDGE_SOURCE)
    ctx = grafial.run_flow(program, "TestFlow")
    g = ctx.get_graph("out")

    nodes = g.nodes("Entity")
    assert len(nodes) == 2
    node = nodes[0]
    assert isinstance(node.id, str)
    assert node.label == "Entity"
    assert node.has_attr("value")
    assert isinstance(node.E("value"), float)
    assert isinstance(node.Var("value"), float)
    assert "NodeView(" in repr(node)

    with pytest.raises(RuntimeError, match="missing attr"):
        node.E("missing")
    with pytest.raises(RuntimeError, match="missing attr"):
        node.Var("missing")

    edges = g.edges("CONNECTED")
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge.src, str)
    assert isinstance(edge.dst, str)
    assert edge.type == "CONNECTED"
    assert edge.is_independent()
    assert not edge.is_competing()
    assert edge.forced_state == "absent"
    assert edge.prob < 1e-3
    assert "EdgeView(" in repr(edge)


def test_pandas_export_raises_clear_error_when_dependency_missing(monkeypatch):
    program = grafial.compile(FORCED_EDGE_SOURCE)
    ctx = grafial.run_flow(program, "TestFlow")
    g = ctx.get_graph("out")

    _block_import(monkeypatch, "pandas")
    with pytest.raises(RuntimeError, match="pandas is not installed"):
        g.to_pandas()


def test_networkx_export_raises_clear_error_when_dependency_missing(monkeypatch):
    program = grafial.compile(FORCED_EDGE_SOURCE)
    ctx = grafial.run_flow(program, "TestFlow")
    g = ctx.get_graph("out")

    _block_import(monkeypatch, "networkx")
    with pytest.raises(RuntimeError, match="networkx is not installed"):
        g.to_networkx()
