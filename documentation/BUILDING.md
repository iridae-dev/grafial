# Building Grafial

This document describes how to build and install the Grafial engine and CLI tool.

## Prerequisites

- **Rust**: Version 1.70 or later (install from [rustup.rs](https://rustup.rs/))
- **Cargo**: Included with Rust installation
- **Nix** (optional): For development environment with pinned toolchains (see `shell.nix`)

## Building the Workspace

Grafial is organized as a Cargo workspace with multiple crates. Build all crates:

```bash
cargo build --workspace --release
```

Build a specific crate:

```bash
# Build core library
cargo build -p grafial-core --release

# Build CLI tool
cargo build -p grafial-cli --release

# Build Python bindings
cargo build -p grafial-python --release
```

## Building the CLI Tool

The Grafial CLI is in the `grafial-cli` crate.

### Development Build

```bash
cargo build -p grafial-cli
```

The binary will be at `target/debug/grafial`.

### Release Build

```bash
cargo build -p grafial-cli --release
```

The optimized binary will be at `target/release/grafial`.

### Installation

Install the CLI tool system-wide using Cargo:

```bash
cargo install --path crates/grafial-cli
```

This installs `grafial` to `~/.cargo/bin/` (or `$CARGO_HOME/bin` if set). Make sure this directory is in your `PATH`.

### Installation from Git

To install directly from a git repository:

```bash
cargo install --git <repository-url> --path crates/grafial-cli
```

## CLI Usage

Once installed, the `grafial` command is available:

```bash
# Validate a Grafial program
grafial crates/grafial-examples/social.grafial

# List all flows in a program
grafial crates/grafial-examples/social.grafial --list-flows

# Execute a specific flow
grafial crates/grafial-examples/social.grafial --flow Demo

# Output results as JSON
grafial crates/grafial-examples/social.grafial --flow Demo -o json

# Get detailed debug output
grafial crates/grafial-examples/social.grafial --flow Demo -o debug

# Report canonical-style compatibility forms
grafial crates/grafial-examples/social.grafial --lint-style

# Rewrite file in-place to canonical style
grafial crates/grafial-examples/social.grafial --fix-style
```

### Command-Line Options

- `<FILE>`: Input `.grafial` file (required)
- `-f, --flow <NAME>`: Flow name to execute (optional - just validates if not provided)
- `-o, --output <FORMAT>`: Output format: `summary` (default), `json`, or `debug`
- `-l, --list-flows`: List all flows in the program instead of executing
- `--lint-style`: Report compatibility syntax that should be modernized to canonical style
- `--fix-style`: Rewrite compatibility syntax in-place to canonical style
- `-h, --help`: Print help information
- `-V, --version`: Print version information

## Testing

Run all tests in the workspace:

```bash
cargo test --workspace
```

Run tests for a specific crate:

```bash
# Test core engine
cargo test -p grafial-core

# Test frontend (parser, AST, validation)
cargo test -p grafial-frontend

# Test integration tests
cargo test -p grafial-tests
```

Run tests with verbose output:

```bash
RUST_LOG=debug cargo test --workspace -- --nocapture
```

Run a specific test:

```bash
cargo test -p grafial-tests --test integration_tests -- parses_social_example
```

### Phase 6 Release Gate

Run the same hardening checks used by CI:

```bash
./scripts/phase6_release_gate.sh
```

## Benchmarks

Run performance benchmarks:

```bash
cargo bench -p grafial-benches
```

## Development Environment

### Using Nix (Recommended)

If you have Nix installed, use the provided shell for a consistent development environment:

```bash
nix-shell
```

This provides:
- Rust toolchain (pinned version)
- Python toolchain (for Python bindings)
- `PYO3_PYTHON` environment variable set correctly

### Without Nix

Ensure you have:
- Rust 1.70+ installed via `rustup`
- Standard build tools for your platform

## Features

`grafial-core` supports optional features:

- `serde`: Enable serialization support in `grafial-core`
- `tracing`: Enable structured logging
- `rayon`: Enable parallel execution (experimental)
- `bincode`: Enable binary serialization

Build with features:

```bash
# Build core with all features
cargo build -p grafial-core --release --features serde,tracing,rayon,bincode
```

`grafial-cli` already depends on `grafial-core` with `serde` enabled, so normal CLI builds include JSON output support by default.

## Troubleshooting

### CLI not found after installation

Ensure `~/.cargo/bin` is in your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Add this to your shell configuration file (`.bashrc`, `.zshrc`, etc.) to make it permanent.

### JSON output not working

Workspace builds of `grafial-cli` include JSON support by default. If you are building custom binaries, ensure `grafial-core` is compiled with `serde`.

### Build errors

- Ensure you're using Rust 1.70 or later: `rustc --version`
- Try cleaning and rebuilding: `cargo clean && cargo build --workspace`
- Check that all dependencies are available: `cargo update`

### Workspace build issues

If you encounter issues building the workspace:

```bash
# Clean all build artifacts
cargo clean

# Update dependencies
cargo update

# Rebuild from scratch
cargo build --workspace --release
```

## Python Bindings

### Prerequisites

- Python 3.8 or later
- [maturin](https://github.com/PyO3/maturin) (install via `pip install maturin` or `cargo install maturin`)
- Or use [uv](https://github.com/astral-sh/uv) for Python environment management

### Development Installation

**Using uv (recommended):**

```bash
cd crates/grafial-python
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Using maturin:**

```bash
cd crates/grafial-python
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release
```

### Running Python Tests

```bash
cd crates/grafial-python
pytest tests/
```

### Building for Distribution

Build wheels for distribution:

```bash
cd crates/grafial-python
maturin build --release
```

This creates `.whl` files in `dist/` that can be installed with `pip install dist/grafial-*.whl`.

### Using in Python Projects

Once installed, import and use:

```python
import grafial

program = grafial.compile("...")
ctx = grafial.run_flow(program, "MyFlow")
```

See `crates/grafial-python/README.md` for detailed documentation.

## Next Steps

- See `LANGUAGE_GUIDE.md` for Grafial syntax and semantics
- See `ENGINE_ARCHITECTURE.md` for engine internals
- See `crates/grafial-python/README.md` for Python bindings documentation
- Check `crates/grafial-examples/` for sample Grafial programs
