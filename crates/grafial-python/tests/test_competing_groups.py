import grafial
from pathlib import Path


def load_example(path: str) -> str:
    # Find project root: tests/ is in crates/grafial-python/, so go up 2 levels
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent
    example_path = project_root / path
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read()


def test_competing_groups_routing_pipeline():
    source = load_example("crates/grafial-examples/competing_choices.grafial")
    prog = grafial.compile(source)
    ctx = grafial.run_flow(prog, "RoutingPipeline")
    g = ctx.get_graph("final_routing")
    assert g is not None

    groups = g.competing_groups("ROUTES_TO")
    assert len(groups) >= 1

    any_has_winner_or_high_entropy = False
    for grp in groups:
        assert isinstance(grp.source_node, str) or isinstance(grp.source_node, int)
        assert grp.edge_type == "ROUTES_TO"
        assert len(grp.categories) >= 1
        assert len(grp.probabilities) == len(grp.categories)
        assert grp.entropy >= 0.0
        if grp.winner(0.01) is not None or grp.entropy > 0.0:
            any_has_winner_or_high_entropy = True

    assert any_has_winner_or_high_entropy

    # Also ensure edges include competing types
    edges = g.edges("ROUTES_TO")
    assert any(e.is_competing() for e in edges)

