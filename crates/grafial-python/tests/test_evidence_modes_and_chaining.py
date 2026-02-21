import grafial

from _helpers import load_example


COMPETING_SOURCE = """
schema Routing {
  node Router { }
  edge ROUTES_TO { }
}

belief_model RouteBeliefs on Routing {
  node Router { }
  edge ROUTES_TO {
    exist ~ Categorical(group_by=source, prior=uniform, pseudo_count=1.0)
  }
}

evidence BaseEvidence on RouteBeliefs {
  choose edge ROUTES_TO(Router["A"], Router["B"])
  choose edge ROUTES_TO(Router["A"], Router["C"])
}

flow RouteFlow on RouteBeliefs {
  graph g = from_evidence BaseEvidence
  export g as "out"
}
"""


def _single_group_snapshot(graph):
    groups = graph.competing_groups("ROUTES_TO")
    assert len(groups) == 1
    group = groups[0]
    probs = group.prob_vector()
    assert abs(sum(probs) - 1.0) < 1e-12
    return group


def test_runtime_competing_evidence_modes_reduce_entropy():
    program = grafial.compile(COMPETING_SOURCE)

    baseline_graph = grafial.run_flow(program, "RouteFlow").get_graph("out")
    baseline_group = _single_group_snapshot(baseline_graph)
    baseline_entropy = baseline_group.entropy
    baseline_probs = baseline_group.prob_vector()

    chosen = grafial.Evidence("Chosen", model="RouteBeliefs")
    chosen.observe_edge_chosen("Router", "A", "ROUTES_TO", "Router", "B")
    chosen_group = _single_group_snapshot(
        grafial.run_flow_with_evidence(program, "RouteFlow", chosen).get_graph("out")
    )
    assert chosen_group.entropy < baseline_entropy

    unchosen = grafial.Evidence("Unchosen", model="RouteBeliefs")
    unchosen.observe_edge_unchosen("Router", "A", "ROUTES_TO", "Router", "B")
    unchosen_group = _single_group_snapshot(
        grafial.run_flow_with_evidence(program, "RouteFlow", unchosen).get_graph("out")
    )
    assert any(
        abs(a - b) > 1e-12
        for a, b in zip(unchosen_group.prob_vector(), baseline_probs)
    )

    forced = grafial.Evidence("Forced", model="RouteBeliefs")
    forced.observe_edge_forced_choice("Router", "A", "ROUTES_TO", "Router", "B")
    forced_group = _single_group_snapshot(
        grafial.run_flow_with_evidence(program, "RouteFlow", forced).get_graph("out")
    )
    assert forced_group.entropy < chosen_group.entropy
    assert max(forced_group.prob_vector()) > 0.99
    assert forced_group.winner(0.001) is not None


def test_runtime_evidence_clear_and_optional_prior_context_path():
    source = load_example("crates/grafial-examples/minimal.grafial")
    program = grafial.compile(source)

    base_graph = grafial.run_flow(program, "MinimalFlow").get_graph("output")
    base_node_count = len(base_graph.nodes())

    ev = grafial.Evidence("Runtime", model="MinimalBeliefs")
    ev.observe_numeric("Entity", "C", "value", 2.0)
    added_graph = grafial.run_flow_with_evidence(program, "MinimalFlow", ev).get_graph("output")
    assert len(added_graph.nodes()) > base_node_count

    ev.clear()
    cleared_graph = grafial.run_flow_with_evidence(program, "MinimalFlow", ev).get_graph("output")
    assert len(cleared_graph.nodes()) == base_node_count

    pipeline = grafial.compile(load_example("crates/grafial-examples/pipeline_composition.grafial"))
    ctx1 = grafial.run_flow(pipeline, "CleaningStage")
    ev2 = grafial.Evidence("Runtime", model="PipelineBeliefs")
    ev2.observe_numeric("Record", "R100", "raw_value", 15.0)
    ctx2 = grafial.run_flow_with_evidence(pipeline, "EnrichmentStage", ev2, ctx1)
    assert ctx2.get_graph("enriched_graph") is not None
