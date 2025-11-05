# Repository Guidelines

## Project Structure & Module Organization
- Core (planned): `src/frontend/` (parser + AST), `src/ir/`, `src/engine/` (graph, rules, flows), `src/metrics/`, `src/storage/`, `src/bindings/` (PyO3).  
- DSL grammar: `grammar/baygraph.pest`.  
- Examples: `examples/*.bg`.  
- Tests: unit tests alongside modules; integration tests in `tests/`.  
- Dev shell: `shell.nix` provides Rust + Python toolchains.

## Build, Test, and Development Commands
- Build: `cargo build` (release: `cargo build --release`)  
- Test: `cargo test` (verbose: `RUST_LOG=debug cargo test -q`)  
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`  
- Format check: `cargo fmt --all -- --check`  
- Python wheel/dev install: `maturin develop --release` (inside `nix-shell`)

## Coding Style & Naming Conventions
- Rust edition: stable; format with `rustfmt` defaults; fix all Clippy warnings.  
- Naming: modules/dirs `snake_case`; types/traits `CamelCase`; fns/vars `snake_case`; consts `SCREAMING_SNAKE_CASE`.  
- Errors: return `Result<T, ExecError>`; avoid panics in library code.  
- Determinism: iterate in stable ID order; no reliance on map/hash iteration order.  
- Prefer immutable graphs between transforms; use `Arc` + copy-on-write for deltas.

## Testing Guidelines
- Unit tests co-located with modules; integration tests in `tests/` (e.g., `tests/parser_tests.rs`).  
- Add property tests with `proptest` for posterior invariants when practical.  
- Name tests with intent (e.g., `beta_updates_on_present_absent`).  
- Include tests for new metrics, rule behaviors, and determinism.  
- Run: `cargo test` locally before pushing.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., `feat(engine): add metric registry`, `fix(frontend): reject invalid prob() targets`).  
- PRs: include description, rationale, linked issues, and tests; note performance impact and add benches if relevant.  
- Pre-merge checklist: build, tests, `clippy`, `fmt`, docs updated (`baygraph_design.md`, `baygraph_roadmap.md`).

## Security & Configuration Tips
- Use the Nix shell to pin toolchains; `PYO3_PYTHON` is set for PyO3.  
- Don’t commit secrets; prefer env vars and local configs.  
- Feature flags: consider `tracing` and `rayon` as opt-in; default to deterministic, single-threaded execution.

## Agent-Specific Instructions
- Follow this file for style and structure.  
- Scope changes minimally; avoid unrelated edits.  
- When touching public APIs, update design docs and tests in the same change.
