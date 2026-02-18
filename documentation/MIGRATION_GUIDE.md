# Migration Guide (Phase 8)

This guide covers migration from removed legacy action syntax to canonical Grafial syntax.

## Legacy -> Canonical Mapping

- `set_expectation A.attr = expr`
  - `non_bayesian_nudge A.attr to expr variance=preserve`
- `force_absent e`
  - `delete e confidence=high`
- `A.attr ~= expr (precision=..., count=...)`
  - `A.attr ~= expr precision=... count=...`
- `delete e (confidence=...)`
  - `delete e confidence=...`
- `suppress e (weight=...)`
  - `suppress e weight=...`

## Auto-Fix Coverage

- CLI:
  - `grafial path/to/file.grafial --fix-style`
  - rewrites all canonical-style compatibility forms above
- LSP quick fix:
  - surfaces canonical style diagnostics with one-click replacement

Stable canonical style lint codes used by tooling:

- `canonical_set_expectation`
- `canonical_force_absent`
- `canonical_inline_args`

## Recommended Migration Flow

1. Run `grafial <file> --fix-style`.
2. Run `grafial <file> --lint-style` and ensure zero canonical-style warnings.
3. Run `cargo test -p grafial-tests --test phase6_release_gate` to confirm examples/flows still pass.
