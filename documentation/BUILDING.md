# Building Grafial

This document describes how to build and install the Grafial engine and CLI tool.

## Prerequisites

- **Rust**: Version 1.70 or later (install from [rustup.rs](https://rustup.rs/))
- **Cargo**: Included with Rust installation
- **Nix** (optional): For development environment with pinned toolchains (see `shell.nix`)

## Building the Library

To build the Grafial library:

```bash
cargo build --release
```

This produces a release-optimized library in `target/release/libgrafial.rlib`.

## Building the CLI Tool

The Grafial CLI is included as a binary target in the main crate.

### Development Build

```bash
cargo build --bin grafial
```

The binary will be at `target/debug/grafial`.

### Release Build

```bash
cargo build --release --bin grafial
```

The optimized binary will be at `target/release/grafial`.

### Installation

Install the CLI tool system-wide using Cargo:

```bash
cargo install --path .
```

This installs `grafial` to `~/.cargo/bin/` (or `$CARGO_HOME/bin` if set). Make sure this directory is in your `PATH`.

### Installation from Git

To install directly from a git repository:

```bash
cargo install --git <repository-url>
```

## CLI Usage

Once installed, the `grafial` command is available:

```bash
# Validate a Grafial program
grafial examples/social.grafial

# List all flows in a program
grafial examples/social.grafial --list-flows

# Execute a specific flow
grafial examples/social.grafial --flow Demo

# Output results as JSON (requires serde feature)
grafial examples/social.grafial --flow Demo -o json

# Get detailed debug output
grafial examples/social.grafial --flow Demo -o debug
```

### Command-Line Options

- `<FILE>`: Input `.grafial` file (required)
- `-f, --flow <NAME>`: Flow name to execute (optional - just validates if not provided)
- `-o, --output <FORMAT>`: Output format: `summary` (default), `json`, or `debug`
- `-l, --list-flows`: List all flows in the program instead of executing
- `-h, --help`: Print help information
- `-V, --version`: Print version information

## Testing

Run all tests:

```bash
cargo test
```

Run tests with verbose output:

```bash
RUST_LOG=debug cargo test -- --nocapture
```

Run only integration tests:

```bash
cargo test --test integration_tests
```

## Development Environment

### Using Nix (Recommended)

If you have Nix installed, use the provided shell for a consistent development environment:

```bash
nix-shell
```

This provides:
- Rust toolchain (pinned version)
- Python toolchain (for future Python bindings)
- `PYO3_PYTHON` environment variable set correctly

### Without Nix

Ensure you have:
- Rust 1.70+ installed via `rustup`
- Standard build tools for your platform

## Features

Grafial supports optional features that can be enabled during build:

- `serde`: Enable serialization support (required for JSON output in CLI)
- `tracing`: Enable structured logging
- `rayon`: Enable parallel execution (experimental)

Build with features:

```bash
cargo build --release --features serde,rayon
```

## Troubleshooting

### CLI not found after installation

Ensure `~/.cargo/bin` is in your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Add this to your shell configuration file (`.bashrc`, `.zshrc`, etc.) to make it permanent.

### JSON output not working

JSON output requires the `serde` feature. Build with:

```bash
cargo build --release --features serde
cargo install --path . --features serde
```

### Build errors

- Ensure you're using Rust 1.70 or later: `rustc --version`
- Try cleaning and rebuilding: `cargo clean && cargo build`
- Check that all dependencies are available: `cargo update`

## Next Steps

- See `LANGUAGE_GUIDE.md` for Grafial syntax and semantics
- See `ENGINE_ARCHITECTURE.md` for engine internals
- See `PYTHON_PLAN.md` for Python bindings roadmap
- Check `examples/` for sample Grafial programs

