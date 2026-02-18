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
- Code actions (quick fixes):
  - Rewrite compatibility forms to canonical inline arguments
  - Wrap bare uncertain field accesses with `E[...]` when validation requests explicit wrappers

## VS Code Integration

Use the `crates/grafial-vscode` extension. Configure `grafial.serverPath` to point to the built binary or add it to PATH.
