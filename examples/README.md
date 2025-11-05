# Baygraph Examples

These examples showcase the unique strengths of Baygraph: probabilistic reasoning, uncertainty propagation, and Bayesian updates over graph structures.

## Core Concepts Demonstrated

### 1. `uncertainty_propagation.bg`
**Key Concept:** Bayesian updates accumulate evidence and preserve uncertainty

**What makes it unique:**
- Shows how repeated evidence strengthens beliefs (e.g., multiple observations of the same edge)
- Demonstrates reasoning with incomplete information (can reason about edges before observing them)
- Soft probabilistic thresholds (e.g., "flag if P(suspicious) > 0.3") vs hard boolean decisions
- Uncertainty is preserved throughout - not eliminated by updates

**Why this is hard in other languages:**
- Most graph languages require exact facts (edge exists or doesn't)
- Baygraph reasons about probabilities, not just boolean facts
- Updates are Bayesian (preserve uncertainty), not deterministic assignments

---

### 2. `competing_choices.bg`
**Key Concept:** Mutually exclusive choices where probabilities must sum to 1

**What makes it unique:**
- Uses `CategoricalPosterior` for routing decisions (each router routes to exactly ONE destination)
- Demonstrates `winner()` function for deterministic decisions when there's a clear winner
- Shows `entropy()` for detecting uncertainty (no clear winner)
- Dynamic category discovery: new destinations added automatically when observed

**Why this is hard in other languages:**
- Most languages model choices as independent edges (probabilities don't sum to 1)
- Baygraph enforces the constraint that exactly one choice is made (probabilities sum to 1)
- This is the correct Bayesian model for mutually exclusive choices

---

### 3. `probabilistic_pattern_matching.bg`
**Key Concept:** Pattern matching based on probability distributions, not hard facts

**What makes it unique:**
- Rules match patterns with soft thresholds (e.g., "if P(friends) >= 0.7")
- Uses `exists` subqueries for probabilistic reachability queries
- Shows how repeated evidence strengthens beliefs
- Demonstrates indirect influence propagation through uncertain paths

**Why this is hard in other languages:**
- Traditional graph queries return exact matches (edge exists or doesn't)
- Baygraph matches probabilistically - accounts for uncertainty in matches
- Can reason about indirect connections even when direct evidence is uncertain

---

### 4. `soft_vs_hard_updates.bg`
**Key Concept:** Soft Bayesian updates preserve uncertainty vs hard constraints that eliminate it

**What makes it unique:**
- `set_expectation` updates the mean but preserves precision (uncertainty)
- Demonstrates the difference between soft updates and hard constraints
- Shows how `force_absent` sets probability to near-zero (but not exactly zero)
- Updates are reversible if new evidence comes in

**Why this is hard in other languages:**
- Most languages use hard assignments (x = value) which eliminate uncertainty
- Baygraph's soft updates preserve the ability to update beliefs later
- This is critical for probabilistic reasoning - you never want to eliminate uncertainty prematurely

---

### 5. `social.bg` (original example)
**Key Concept:** Basic multi-pattern rules with value transfers

**What makes it unique:**
- Shows simple pattern matching across multiple edges
- Demonstrates value transfers between nodes
- Uses `force_absent` to disconnect edges

---

## What Makes Baygraph Different

### 1. **Probabilistic, Not Deterministic**
Most graph languages assume edges either exist or don't. Baygraph models edges as probability distributions, allowing you to:
- Reason about uncertain relationships
- Accumulate evidence over time
- Make decisions based on probability thresholds

### 2. **Bayesian Updates Preserve Uncertainty**
Unlike hard database updates, Baygraph's updates preserve uncertainty:
- `set_expectation` shifts the mean but keeps the precision
- `force_absent` sets probability to near-zero but doesn't eliminate it
- New evidence can always update beliefs

### 3. **Competing Choices with Proper Constraints**
When choices are mutually exclusive (like routing), Baygraph enforces that probabilities sum to 1:
- Uses Dirichlet-Categorical posterior (not independent Bernoulli)
- Automatically normalizes probabilities
- Supports dynamic category discovery

### 4. **Soft Pattern Matching**
Rules match patterns probabilistically:
- Thresholds are soft (e.g., "if P >= 0.7")
- Accounts for uncertainty in matches
- Can reason about indirect connections

### 5. **Uncertainty-Aware Metrics**
Metrics understand uncertainty:
- Can compute expected values over distributions
- Can reason about probabilities in aggregations
- Support probabilistic filtering

---

## Running Examples

```bash
# Parse and validate an example
cargo run --bin print_ast -- examples/uncertainty_propagation.bg

# Run through the full engine (requires flow execution)
# (Python bindings coming soon)
```

---

## Next Steps

After understanding these examples, you should be able to:
1. Model uncertain relationships as probability distributions
2. Use Bayesian updates to accumulate evidence
3. Write rules that reason probabilistically
4. Handle competing choices with proper constraints
5. Compute metrics over uncertain graphs

See `LANGUAGE_GUIDE.md` for complete language reference.

