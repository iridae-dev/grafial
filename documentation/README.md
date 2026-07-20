# Documentation Index

This folder contains contributor and user-facing documentation for Grafial.

## Documents

- [BUILDING.md](BUILDING.md)
  - Build/install instructions for workspace crates, CLI usage, tests, Python bindings, and the WebAssembly package.
- [LANGUAGE_GUIDE.md](LANGUAGE_GUIDE.md)
  - DSL syntax, semantics, canonical style, and examples.
- [PROBABILISTIC_SEMANTICS.md](PROBABILISTIC_SEMANTICS.md)
  - Normative posterior-update equations, `infer_beliefs` policy, AIC/BIC.
- [EXAMPLES.md](EXAMPLES.md)
  - Index of repository examples (problem, concepts, flows, commands).
- [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md)
  - Internal architecture for frontend, IR, core engine, storage/runtime model, and extension points.
- [crates/grafial-wasm/README.md](../crates/grafial-wasm/README.md)
  - Browser/WebAssembly JSON API for parsing, inspecting, and executing Grafial programs.
- [webapp/README.md](../webapp/README.md)
  - Grafial Composer ([hosted](https://grafial.iridae.com/)): in-browser visual editor.

## Where to Start

- New users:
  - Start with the [root README](../README.md), then try [the Composer](https://grafial.iridae.com/).
  - Then read the [language guide](LANGUAGE_GUIDE.md) and [examples index](EXAMPLES.md).
- Contributors:
  - Start with [CONTRIBUTING.md](../CONTRIBUTING.md).
  - Then read [BUILDING.md](BUILDING.md), [PROBABILISTIC_SEMANTICS.md](PROBABILISTIC_SEMANTICS.md), and [ENGINE_ARCHITECTURE.md](ENGINE_ARCHITECTURE.md).

## Current State

Compiler/runtime milestone phases are complete through the current implementation baseline.
Operational and architectural guidance is now maintained directly in these living documents
rather than a separate roadmap file.
