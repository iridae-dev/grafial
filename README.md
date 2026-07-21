[![All Tests](https://github.com/iridae-dev/grafial/actions/workflows/phase6-release-gate.yml/badge.svg)](https://github.com/iridae-dev/grafial/actions/workflows/phase6-release-gate.yml) [![Release](https://img.shields.io/github/v/release/iridae-dev/grafial)](
  https://github.com/iridae-dev/grafial/releases/latest)
  
# Grafial

**Model graphs where connections are hypotheses, not facts.**

Grafial is a declarative language and runtime for reasoning over uncertain graphs. Use it when your domain is naturally graph-shaped, but nodes, attributes, and relationships cannot honestly be represented as fixed facts. Grafial lets you define prior beliefs, update them as evidence arrives, match graph patterns probabilistically, and assemble the resulting rules and analyses into deterministic, auditable flows.

Instead of maintaining a graph alongside a separate collection of confidence scores, Bayesian updates, threshold rules, and pipeline scripts, you describe them together in a version-controlled .grafial program.

- [Try Grafial in the browser](https://grafial.iridae.com/)
- `pip install grafial`
- [Download the CLI](https://github.com/iridae-dev/grafial/releases)

## Is Grafial for your problem?

Grafial may be a good fit when:

- the existence of a relationship is uncertain or disputed;
- evidence arrives repeatedly and should update—not overwrite—previous beliefs;
- relationships compete with one another, such as mutually exclusive routes or choices;
- graph rules should act only when a probabilistic condition is sufficiently credible;
- you need to reproduce and inspect how evidence, inference, and rules produced a result.

Typical domains might include relationship assessment, entity-link hypotheses, fraud or risk networks, routing decisions, trust graphs, dependency analysis, and other systems where an edge is better understood as a belief than a boolean.

Grafial is probably not the right tool when:

- your graph is already known and deterministic;
- you primarily need persistent graph storage or high-volume graph queries;
- you need arbitrary probabilistic models or unrestricted probabilistic programming;
- a conventional dataframe, graph library, or rules engine already expresses the problem clearly.

## See it in one minute

Open the [Grafial Composer](https://grafial.iridae.com/), select:

`Examples → probabilistic_pattern_matching → Run`

That example starts with uncertain claims about friendships and influence:

```grafial
edge FRIENDS {
  exist ~ Bernoulli(prior=0.5, weight=2.0)
}
```

Evidence can repeat or conflict:

```grafial
FRIENDS(Person -> Person) {
  "Alice" -> "Bob";
  "Alice" -> "Bob";
  "Alice" -> "Bob";

  "Carol" -/> "Alice";
  "Carol" -/> "Alice";
  "Carol" -/> "Alice"
}
```

Grafial accumulates those observations into posterior beliefs. A rule can then match only relationships that are sufficiently credible:

```grafial
rule PropagateInfluence on SocialBeliefs {
  pattern
    (A:Person)-[friend:FRIENDS]->(B:Person)

  where
    prob(friend) >= 0.7
    and E[A.influence] > E[B.influence]

  action {
    let transfer = (E[A.influence] - E[B.influence]) * 0.1

    non_bayesian_nudge B.influence
      to E[B.influence] + transfer
      variance=preserve
  }
}
```

The relationship never has to become an artificial true or false. Rules can reason directly about its posterior probability.

A flow makes the analysis reproducible:

```grafial
flow SocialAnalysis on SocialBeliefs {
  graph observed = from_evidence SocialEvidence
  graph updated = observed |> apply_rule PropagateInfluence

  metric strong_connections on updated =
    avg_degree(Person, FRIENDS, min_prob=0.7)

  export updated as "influence_graph"
}
```

The result includes the posterior graph, calculated metrics, rule-firing audit information, and inference diagnostics.

## The Grafial mental model

A program is built from five kinds of declaration:

| Declaration      | Purpose     |
| ------------- | ------------- |
| schema | Defines typed nodes, attributes, and edges. |
| belief_model | Defines priors and posterior families. |
| evidence	| Adds observations, including repeated and conflicting observations. |
| rule	| Matches graph patterns and applies probabilistic or deterministic actions. |
| flow	| Builds and transforms graphs, computes metrics, and exports results. |

For example:

```grafial
schema Network {
  node Entity {
    risk: Real
  }

  edge LINKED { }
}

belief_model NetworkBeliefs on Network {
  node Entity {
    risk ~ Gaussian(mean=0.0, precision=0.1)
  }

  edge LINKED {
    exist ~ Bernoulli(prior=0.1, weight=10.0)
  }
}
```

Grafial represents the resulting state as a belief graph: node attributes and relationships carry posterior distributions that can be inspected, queried, transformed, and exported.

## What Grafial supports

Grafial currently includes:

- Gaussian posterior beliefs for node attributes and continuous edge weights;
- Beta/Bernoulli beliefs for independent edge existence;
- Dirichlet/categorical beliefs for competing outgoing edges;
- repeated and precision-weighted observations;
- probabilistic graph-pattern matching;
- expectation, probability, credible-event, entropy, variance, and interval queries;
- deterministic rule execution, including fixpoint rules;
- graph flows with pruning, metrics, snapshots, model selection, and exports;
- scoped belief propagation for related independent edges;
- intervention audits and inference diagnostics;
- a CLI, Python bindings, Rust APIs, WebAssembly bindings, an LSP, and a browser-based visual Composer.

Grafial is alpha software. Its probabilistic semantics are intentionally explicit and limited rather than pretending to be a general-purpose inference system. 
See the [language guide](documentation/LANGUAGE_GUIDE.md) and
[normative probabilistic semantics](documentation/PROBABILISTIC_SEMANTICS.md)
for the precise behavior.

## Install

### Python

```bash
pip install grafial
```

```python
import pathlib
import grafial

source = pathlib.Path("analysis.grafial").read_text()
program = grafial.compile(source)
result = grafial.run_flow(program, "Analysis")

print(result.metrics)
print(result.inference_diagnostics)
```

### CLI

Download the appropriate archive from [GitHub Releases](https://github.com/iridae-dev/grafial/releases), extract it, and place grafial on your `PATH`.

```
grafial analysis.grafial --list-flows
grafial analysis.grafial --flow Analysis
grafial analysis.grafial --flow Analysis --output json
```

### Browser

The [Grafial Composer](https://grafial.iridae.com/) runs the WebAssembly engine entirely in the browser. It includes structured editors for schemas, belief models, evidence, rules, and flows, plus posterior graph visualization, metrics, rule audits, and inference diagnostics.

### Explore the examples

The repository includes examples covering:

- probabilistic graph-pattern matching;
- fraud-risk reasoning under incomplete evidence;
- competing routing choices;
- prior sensitivity;
- soft versus hard updates;
- graph pipelines and metric sharing;
- uncertainty propagation;
- A/B decision analysis.

Start with the [examples index](documentation/EXAMPLES.md).
