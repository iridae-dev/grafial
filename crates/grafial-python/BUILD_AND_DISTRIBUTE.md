# Building and Distributing Grafial Python Bindings

## Overview

Grafial Python bindings are built using [maturin](https://maturin.rs/), which compiles Rust code into a Python extension module (`.so` on Linux/macOS, `.pyd` on Windows). The package can be installed locally, distributed as wheels, or published to PyPI.

## Build Tools

### Maturin

Maturin is the standard tool for building PyO3-based Python packages. It:
- Compiles Rust code to a Python extension module
- Generates Python package metadata
- Creates distributable wheels
- Handles cross-compilation for multiple platforms

### UV (Optional)

UV is a fast Python package manager that can also build PyO3 packages. It uses maturin under the hood but provides a simpler interface.

## Local Development

### 1. Development Install (Editable)

This installs the package in "editable" mode so you can modify Rust code and rebuild without reinstalling.

**Using uv:**
```bash
cd crates/grafial-python
uv venv
source .venv/bin/activate
uv pip install -e .
```

**Using maturin:**
```bash
cd crates/grafial-python
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

### 2. Rebuilding After Changes

After modifying Rust code (`src/lib.rs`), rebuild:

```bash
# With uv
uv pip install -e . --force-reinstall

# With maturin
maturin develop --release
```

### 3. Running Tests

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v
```

## Building for Distribution

### Building Wheels

Wheels (`.whl` files) are the standard Python package format. They contain pre-built binaries for specific platforms.

**Build for current platform:**
```bash
maturin build --release
```

This creates wheels in `dist/` directory:
- `grafial-0.1.0-cp38-cp38-macosx_10_9_x86_64.whl` (example)
- `grafial-0.1.0-cp39-cp39-macosx_10_9_x86_64.whl`
- etc.

**Build for multiple platforms:**
```bash
# Requires cross-compilation setup (see maturin docs)
maturin build --release --target x86_64-unknown-linux-gnu
maturin build --release --target x86_64-pc-windows-msvc
```

### Building Source Distribution

Source distributions (`.tar.gz` files) contain the Rust source code and can be built on any platform:

```bash
maturin sdist
```

This creates `dist/grafial-0.1.0.tar.gz` that can be:
- Built on any platform with Rust installed
- Uploaded to PyPI
- Installed with `pip install grafial-0.1.0.tar.gz`

## Installing from Built Packages

### From Wheel

```bash
pip install dist/grafial-0.1.0-*.whl
```

### From Source Distribution

```bash
pip install dist/grafial-0.1.0.tar.gz
```

This will compile the Rust code during installation (requires Rust toolchain).

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. Generate API tokens in account settings
3. Configure credentials (using `maturin` or `twine`)

### Configure Credentials

**Option 1: Using maturin (recommended)**

```bash
# Set environment variables
export MATURIN_PYPI_TOKEN="pypi-..."
export MATURIN_TEST_PYPI_TOKEN="pypi-..."
```

**Option 2: Using .pypirc**

Create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

### Publish to TestPyPI First

Always test on TestPyPI before publishing to PyPI:

```bash
maturin publish --test-pypi
```

Then test installation:
```bash
pip install -i https://test.pypi.org/simple/ grafial
```

### Publish to PyPI

Once tested, publish to production PyPI:

```bash
maturin publish
```

This will:
1. Build wheels for all supported Python versions
2. Build source distribution
3. Upload to PyPI

### Installing from PyPI

After publishing, users can install:

```bash
pip install grafial
```

## Using in Other Projects

### Option 1: Local Path Installation

```bash
# In your Python project
pip install /path/to/baygraph/crates/grafial-python
```

### Option 2: Git Repository

```bash
pip install git+https://github.com/yourusername/baygraph.git#subdirectory=crates/grafial-python
```

### Option 3: Development Editable Install

For active development on both projects:

```bash
# In your Python project's virtual environment
pip install -e /path/to/baygraph/crates/grafial-python
```

When you modify Rust code:
```bash
cd /path/to/baygraph/crates/grafial-python
maturin develop --release
```

### Option 4: Add as Dependency

In your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "grafial @ git+https://github.com/yourusername/baygraph.git#subdirectory=crates/grafial-python",
]
```

Or after publishing to PyPI:

```toml
[project]
dependencies = [
    "grafial>=0.1.0",
]
```

## Version Management

Update version in `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Update version number
```

Then rebuild and publish:

```bash
maturin build --release
maturin publish
```

## Platform Support

Maturin automatically builds wheels for:
- **macOS**: Intel (x86_64) and Apple Silicon (arm64)
- **Linux**: x86_64 and arm64 (manylinux)
- **Windows**: x86_64 (MSVC)

For other platforms, source distributions can be used (requires Rust toolchain on target platform).

## Continuous Integration

Example GitHub Actions workflow for automated builds:

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    - name: Install maturin
      run: pip install maturin
    - name: Build
      run: |
        cd crates/grafial-python
        maturin build --release
    - name: Test
      run: |
        cd crates/grafial-python
        pip install dist/grafial-*.whl
        pip install pytest
        pytest tests/
```

## Troubleshooting

### Build Issues

- **"Can't find Python"**: Set `PYO3_PYTHON=$(which python3)` or use `--python` flag
- **"Rust not found"**: Install Rust via `rustup`
- **Linker errors**: Install platform-specific build tools (Xcode on macOS, Visual Studio on Windows)

### Distribution Issues

- **Wheels not found for platform**: Build source distribution or set up cross-compilation
- **Import errors after installation**: Ensure correct virtual environment is active
- **Version conflicts**: Use `pip install --upgrade grafial`

## Resources

- [Maturin Documentation](https://maturin.rs/)
- [PyO3 User Guide](https://pyo3.rs/)
- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Publishing Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/#uploading-your-project-to-pypi)

