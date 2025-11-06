# Grafial Python Bindings

Python bindings for Grafial using PyO3 and maturin.

## Quick Start

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.8 or later
- [maturin](https://github.com/PyO3/maturin) (install via `pip install maturin` or `cargo install maturin`)
- Or use [uv](https://github.com/astral-sh/uv) for Python environment management (recommended)

### Development Installation

**Option 1: Using uv (recommended)**

```bash
cd crates/grafial-python

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode (builds and installs the package)
uv pip install -e .
```

**Option 2: Using maturin directly**

```bash
cd crates/grafial-python

# Create virtual environment (or use existing)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install maturin if not already installed
pip install maturin

# Build and install in development mode
maturin develop --release
```

**Option 3: Using nix-shell**

If you're using the project's `shell.nix`:

```bash
# Enter nix shell (from project root)
nix-shell

# Then follow Option 1 or 2 above
cd crates/grafial-python
uv pip install -e .  # or maturin develop --release
```

### Running Tests

After installation, you can run Python tests:

```bash
# From crates/grafial-python directory
pytest tests/

# Or run specific test file
pytest tests/test_basic.py -v
```

You can also run Rust unit tests (these test the bindings from Rust side):

```bash
# From project root
cargo test -p grafial-python
```

### Using in Python

Once installed, you can import and use Grafial:

```python
import grafial

# Compile a Grafial program
source = """
schema Test {
    node Person { }
    edge KNOWS { }
}
belief_model TestBeliefs on Test {
    edge KNOWS { exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0) }
}
"""
program = grafial.compile(source)

# Check what's in the program
print(f"Schemas: {program.get_schema_names()}")
print(f"Models: {program.get_belief_model_names()}")
```

## Building for Distribution

### Building Wheels

To build Python wheels for distribution:

```bash
cd crates/grafial-python

# Build wheels for current platform
maturin build --release

# Build wheels for multiple platforms (requires cross-compilation setup)
maturin build --release --out dist
```

This creates `.whl` files in the `dist/` directory that can be installed with `pip`.

### Installing from Wheel

```bash
pip install dist/grafial-*.whl
```

### Building Source Distribution

```bash
maturin sdist
```

This creates a source distribution that can be built on any platform.

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. Configure credentials (using `twine` or `maturin`)

### Publishing

**1. Test on TestPyPI first:**

```bash
maturin publish --test-pypi
```

**2. Publish to PyPI:**

```bash
maturin publish
```

Maturin will automatically:
- Build wheels for multiple Python versions
- Build source distribution
- Upload to PyPI

### Installing from PyPI (after publishing)

```bash
pip install grafial
```

## Using in Other Python Projects

### Option 1: Install from Local Path

```bash
# In your Python project
pip install /path/to/baygraph/crates/grafial-python
```

### Option 2: Install from Git Repository

```bash
pip install git+https://github.com/yourusername/baygraph.git#subdirectory=crates/grafial-python
```

### Option 3: Add as Dependency in pyproject.toml

```toml
[project]
dependencies = [
    "grafial @ git+https://github.com/yourusername/baygraph.git#subdirectory=crates/grafial-python",
]
```

### Option 4: Development Editable Install

For active development on both projects:

```bash
# In your Python project's virtual environment
pip install -e /path/to/baygraph/crates/grafial-python
```

This installs in "editable" mode, so changes to the Rust code will be reflected after rebuilding (run `maturin develop` again).

## Troubleshooting

### Build Errors

**"Can't find Python"**
- Set `PYO3_PYTHON` environment variable: `export PYO3_PYTHON=$(which python3)`
- Or use `maturin develop --python python3` to specify Python version

**"Missing Rust toolchain"**
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Ensure `cargo` is in your PATH

**"maturin not found"**
- Install maturin: `pip install maturin` or `cargo install maturin`
- Or use `uv pip install maturin` if using uv

### Import Errors

**"ModuleNotFoundError: No module named 'grafial'"**
- Ensure you've installed the package: `pip install -e .` or `maturin develop`
- Check you're using the correct virtual environment
- Verify installation: `pip list | grep grafial`

**"ImportError: dynamic module does not define module export function"**
- Rebuild the extension: `maturin develop --release`
- This usually happens when Rust code changed but wasn't rebuilt

### Testing Issues

**Tests fail with "library not loaded" errors**
- Rebuild: `maturin develop --release`
- Ensure Python version matches the one used to build

## Development Workflow

### Making Changes

1. Edit Rust code in `src/lib.rs`
2. Rebuild: `maturin develop --release` (or `uv pip install -e .`)
3. Test: `pytest tests/`
4. Iterate

### Debugging

For faster iteration during development, use debug builds:

```bash
maturin develop  # Without --release (faster, but slower runtime)
```

For production-like testing:

```bash
maturin develop --release  # Optimized build (slower build, faster runtime)
```

### Code Generation

Maturin automatically:
- Compiles Rust code to a Python extension module
- Generates Python type stubs (`.pyi` files) if configured
- Handles linking and platform-specific details

## Project Structure

```
crates/grafial-python/
├── Cargo.toml          # Rust package configuration
├── pyproject.toml      # Python package configuration (maturin)
├── src/
│   └── lib.rs          # PyO3 bindings code
├── tests/              # Python tests
│   ├── test_basic.py
│   ├── test_flow_minimal.py
│   └── ...
└── README.md           # This file
```

## Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://maturin.rs/)
- [PyO3 User Guide](https://pyo3.rs/latest/)
- [Python Packaging Guide](https://packaging.python.org/)

