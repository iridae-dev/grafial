# Grafial Probabilistic Semantics (Normative)

This document specifies the conjugate update equations and inference policies
implemented by the Grafial runtime. It is the reference for independent
verification. Behavioral prose lives in [LANGUAGE_GUIDE.md](LANGUAGE_GUIDE.md); when the two
diverge, **this file wins for probabilistic numerics**, and bugs should be filed
against the engine.

Notation uses standard Bayesian conjugate forms. All floating-point operations
are IEEE-754 `f64` unless stated otherwise.

---

## 1. Gaussian attributes

Prior: \(p(\mu) = \mathcal{N}(\mu_0, \tau_0^{-1})\) stored as `(mean=μ₀, precision=τ₀)`.

Observation \(x\) with observation precision \(\tau_{\mathrm{obs}} > 0\):

\[
\tau_n = \tau_0 + \tau_{\mathrm{obs}}, \qquad
\mu_n = \frac{\tau_0 \mu_0 + \tau_{\mathrm{obs}} x}{\tau_n}.
\]

Repeated observations apply the same update sequentially (sufficient statistics
add). Soft attribute updates of the form `attr ~= x precision=τ count=c` apply
the observation \(c\) times with precision \(\tau\) (equivalently one update with
precision \(c\tau\) and the same \(x\)).

`non_bayesian_nudge` is **not** a conjugate update; it relocates the mean while
optionally preserving or shrinking variance per the action arguments.

---

## 2. Bernoulli / Beta edge existence

Independent edges store a Beta posterior \((\alpha, \beta)\).

Hard observation `present` / `absent`:

\[
\alpha \leftarrow \alpha + \mathbf{1}[\mathrm{present}], \qquad
\beta \leftarrow \beta + \mathbf{1}[\mathrm{absent}].
\]

Weighted / soft observation with weight \(w \ge 0\):

\[
\alpha \leftarrow \alpha + w\cdot\mathbf{1}[\mathrm{present}], \qquad
\beta \leftarrow \beta + w\cdot\mathbf{1}[\mathrm{absent}].
\]

Prior construction from `Bernoulli(prior=p, weight=W)` / `pseudo_count`:

\[
\alpha_0 = p\cdot W,\qquad \beta_0 = (1-p)\cdot W
\]

(with `weight` and `pseudo_count` treated as the same strength parameter \(W\)).

Mean and variance:

\[
\mathbb{E}[p] = \frac{\alpha}{\alpha+\beta}, \qquad
\mathrm{Var}[p] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.
\]

---

## 3. Categorical / Dirichlet competing edges

`Categorical(group_by=source, ...)` groups edges that share `(source, edge_type)`.
Categories are destination node identities known at evidence-planning time.
`group_by=destination` is **rejected at validation** until implemented.

Uniform prior `prior=uniform, pseudo_count=W` over \(K\) categories:

\[
\alpha_k = W / K \quad (k=1..K).
\]

Explicit prior `prior=[α₁,…,α_K]` requires length \(K\).

Updates:

| Evidence mode | Effect |
|---|---|
| `chosen` / `present` on category \(j\) | \(\alpha_j \leftarrow \alpha_j + 1\) |
| `unchosen` / `absent` on category \(j\) | **Only for \(K=2\)**: equivalent to `observe_chosen(1-j)` (exact conjugate). For \(K>2\), the likelihood \(1-\pi_j\) is **not Dirichlet-conjugate**; the engine returns a validation/execution error asking callers to use an explicit `chosen` category instead. |
| `forced_choice` | Near-deterministic concentration on the chosen category (\(\alpha_j \approx 10^6\), others \(1\)) |

Mean probabilities: \(\mathbb{E}[\pi_k] = \alpha_k / \sum_i \alpha_i\).

---

## 4. `infer_beliefs` (loopy sum-product)

Scope: **independent Beta edges only**. Competing (Dirichlet) edges are left
unchanged by this transform.

### 4.1 Factor graph

Variables = independent edges. Pairwise factors couple edges of the **same edge
type** that share a source (fan-out) or share a destination (fan-in). Chain
adjacency (dst of one = src of another) is **not** coupled.

### 4.2 Messages and coupling

Synchronous loopy sum-product with damping \(d \in [0,1)\). Coupling strength
\(\kappa\) (default `0.6`) scales the pairwise agreement potential; positive
\(\kappa\) favors neighboring edges having similar existence probabilities.

Defaults (`BeliefPropagationConfig::default`):

| Field | Default |
|---|---:|
| `max_iterations` | 32 |
| `damping` | 0.35 |
| `convergence_tolerance` | \(10^{-5}\) |
| `coupling_strength` | 0.6 |

### 4.3 Convergence policy (API contract)

1. Iterate until \(\max|\Delta \mathrm{message}| < \texttt{convergence_tolerance}\)
   or `max_iterations` is reached.
2. **Non-convergence is not an error.** The transform returns the latest graph
   and appends an `inference_diagnostics` event with `converged=false`.
3. Surfaces:
   - Rust: `FlowResult.inference_diagnostics`
   - CLI / JSON / WASM: same field
   - Python: `Context.inference_diagnostics`
4. Callers that require convergence must check `converged` (or raise in their
   own wrapper). Grafial does not hard-fail by default.

Effective sample size of each Beta edge is preserved under message updates
(means are smoothed; pseudo-counts are not discarded).

---

## 5. Model selection (edge AIC / BIC)

Scores are computed from posterior **concentration parameters** (not raw
Bernoulli counts of hard observations).

Independent Beta\((\alpha,\beta)\) edge with \(p=\alpha/(\alpha+\beta)\),
\(q=1-p\):

\[
\ell += \alpha \ln p + \beta \ln q, \qquad k += 1, \qquad n += \alpha+\beta.
\]

Competing Dirichlet group with concentrations \(\alpha_1..\alpha_K\) (each group
counted once):

\[
\ell += \sum_k \alpha_k \ln(\alpha_k / \sum_i \alpha_i), \qquad
k += K-1, \qquad n += \sum_k \alpha_k.
\]

Final scores (lower is better):

\[
\mathrm{AIC} = 2k - 2\ell, \qquad
\mathrm{BIC} = k\ln n - 2\ell
\]

(with \(n\) floored at 1). `select_model { … } by edge_aic|edge_bic` picks the
candidate with the lowest score. Candidates with mismatched effective sample
size are rejected.

---

## 6. Rule fixpoint mode

`mode: fixpoint` repeatedly applies `for_each` semantics until the max absolute
change across comparable posterior parameters is \(< 10^{-6}\) or 1000 iterations
elapse. Non-convergence is an **execution error**.

---

## 7. Verification

Analytical / reference tests live in:

- [bayesian_updates_tests.rs](../crates/grafial-tests/tests/bayesian_updates_tests.rs)
- [probabilistic_golden_tests.rs](../crates/grafial-tests/tests/probabilistic_golden_tests.rs)
- [property_tests.rs](../crates/grafial-tests/tests/property_tests.rs)
