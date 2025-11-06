import grafial


def test_compile_minimal_program():
    source = """
    schema Test { node Person { } edge REL { } }
    belief_model TestBeliefs on Test {
        edge REL { exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0) }
    }
    """
    program = grafial.compile(source)
    assert program is not None
    assert program.get_schema_names() == ["Test"]
    assert program.get_belief_model_names() == ["TestBeliefs"]
    assert program.get_flow_names() == []

