# Grafial: A language for reasoning when connections aren’t clear

Grafial is a domain-specific language for reasoning about **uncertain relationships**.  
It treats graphs not as fixed structures, but as **probabilistic systems** where nodes, edges, and attributes each carry **degrees of belief** rather than binary truth.  

Instead of demanding statistical expertise or hand-written Bayesian math, Grafial lets you **describe uncertain systems declaratively**. You define schemas, evidence, and rules; the engine maintains consistent posterior beliefs automatically.  

---

## 1. Overview

Grafial was built to make it practical to work with **graphs that represent Bayesian uncertainty**: systems where connections, properties, and outcomes are *partially observed, probabilistic, and evolving*.  

Where traditional graph engines deal in facts (“A is connected to B”), Grafial deals in beliefs (“A is probably connected to B, and that affects what we think about C”).  

Example use cases:

- Bayesian A/B testing and decision analysis  
- Probabilistic social or influence networks  
- Causal reasoning and belief propagation  
- Uncertain graph querying and inference

---

## 2. Language

The **Grafial language** defines graph schemas, probabilistic models, evidence updates, and reasoning rules. Its syntax looks familiar to anyone who’s used declarative or rule-based systems, but it’s explicitly designed for uncertainty.  

Key concepts:
- `belief_model` defines priors and posterior distributions for nodes and edges.  
- `evidence` updates beliefs from observations.  
- `rule` expresses probabilistic transformations and reasoning steps.  
- `flow` sequences updates and computes derived metrics.  

Example:

```Grafial
schema ABTest {
  node Variant {
    conversion_rate: Real
  }
  edge OUTPERFORMS { }
}

belief_model TestBeliefs on ABTest {
  node Variant {
    conversion_rate ~ GaussianPosterior(prior_mean=0.1, prior_precision=10.0)
  }
  edge OUTPERFORMS {
    exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0)
  }
}

evidence VariantBData on TestBeliefs {
  observe Variant["B"].conversion_rate = 0.15
}
```

See **`documentation/LANGUAGE_GUIDE.md`** or `examples` for the full grammar and semantics.

---

## 3. Grafial Engine

The **engine** is implemented in Rust and provides the runtime for inference, belief updates, and rule execution. It handles probabilistic pattern matching, posterior updates, and metric computation efficiently, with full type and memory safety.

### CLI Tool

The Grafial CLI allows you to validate and execute Grafial programs from the command line:

```bash
# Validate a program
grafial program.grafial

# List available flows
grafial program.grafial --list-flows

# Execute a flow
grafial program.grafial --flow MyFlow

# Get JSON output
grafial program.grafial --flow MyFlow -o json

# Get debug output
grafial program.grafial --flow MyFlow -o debug
```

See **`documentation/BUILDING.md`** for installation and build instructions.

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

See **`documentation/ENGINE_ARCHITECTURE.md`** for details on the runtime, evaluation model, and API layout.

---

## 4. Python Wrappers

The **Python interface** exposes Grafial’s engine using PyO3 bindings.  It allows you to load, run, and analyze probabilistic graphs interactively: perfect for notebooks, pipelines, or decision systems.

### Install
```bash
cd python
pip install -e .
```

### Example
```python
from Grafial import GraphRuntime

runtime = GraphRuntime()
runtime.load("abtest.grafial")
runtime.run()
```

See **`documentation/PYTHON.md`** for integration details and usage examples.
