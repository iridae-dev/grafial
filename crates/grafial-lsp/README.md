# grafial-lsp

Rust Language Server (LSP) for the Grafial DSL.

## Build

```bash
cargo build -p grafial-lsp --release
```

## Run (stdio)

The server speaks stdio LSP. It is started by the VS Code extension, but you can run it standalone:

```bash
# Prints nothing; waits for LSP on stdio
./target/release/grafial-lsp
```

## Features

- Text synchronization (full)
- Diagnostics:
  - Parse errors with line/column mapping from Pest → LSP
  - Validation errors with semantic source ranges
  - Canonical-style modernization warnings for compatibility syntax
  - Phase 7 statistical guardrail lints with stable codes:
    - `stat_variance_collapse`
    - `stat_prior_dominance`
    - `stat_precision_outlier`
    - `stat_prior_data_conflict`
    - `stat_numerical_instability`
    - `stat_multiple_testing`
    - `stat_circular_update`
    - `stat_delete_explanation`
    - `stat_suppress_explanation`
- Code actions (quick fixes):
  - Rewrite compatibility forms to canonical inline arguments
  - Wrap bare uncertain field accesses with `E[...]` when validation requests explicit wrappers
- Scoped lint suppression pragmas:
  - `// grafial-lint: ignore(<code>)`
  - Applies to the enclosing declaration or next declaration block.

## VS Code Integration

Use the `crates/grafial-vscode` extension. Configure `grafial.serverPath` to point to the built binary or add it to PATH.
