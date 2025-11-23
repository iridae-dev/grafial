# grafial-lsp

Rust Language Server (LSP) for the Grafial DSL. Implements Phase 1 of the roadmap: real-time diagnostics for syntax and validation errors.

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
  - Validation errors (no spans yet) at document start

## VS Code Integration

Use the `crates/grafial-vscode` extension. Configure `grafial.serverPath` to point to the built binary or add it to PATH.
