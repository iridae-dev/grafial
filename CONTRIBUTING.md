# Contributing to Grafial

Thanks for contributing.

## Getting Started

1. Build the workspace:
```bash
cargo build --workspace
```
2. Run checks before opening a PR:
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```
3. If you touch Python bindings:
```bash
cd crates/grafial-python
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest tests -q
```

## Development Expectations

- Keep changes scoped to the task; avoid unrelated edits.
- Follow repository naming/style conventions in `AGENTS.md`.
- Add or update tests for behavior changes.
- Update docs when public APIs, build steps, or features change.

## Pull Requests

PRs should include:
- What changed.
- Why it changed.
- How it was validated (commands + results).
- Any known limitations or follow-ups.

Prefer small, reviewable PRs over large mixed changes.

## Commit Messages

Use imperative, scoped messages where possible, for example:
- `feat(engine): add metric registry`
- `fix(frontend): reject invalid prob() targets`

## Releasing

Push a tag `v*` (e.g. `v0.1.0`) to trigger the release workflow. It builds CLI, LSP, and Python wheels, creates a GitHub Release, and attaches all artifacts. To publish to PyPI automatically, set repository variable `PUBLISH_TO_PYPI` to `true` and add secret `PYPI_API_TOKEN` (from [pypi.org/manage/account/token](https://pypi.org/manage/account/token/)).

## Reporting Bugs

Open a GitHub issue with:
- Reproduction steps
- Expected vs actual behavior
- Environment details (OS, Rust version, command output)

For security issues, follow `SECURITY.md` instead of opening a public issue.

## License

By contributing, you agree that your contributions are licensed under the repository's MIT license.
