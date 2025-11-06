# Quick Start Guide - Grafial Python Bindings

This guide will walk you through building, testing, and using the Grafial Python bindings.

## Step 1: Install Prerequisites

### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Install Python 3.8+ (if not already installed)

Check your Python version:
```bash
python3 --version  # Should be 3.8 or higher
```

### Install uv (recommended) or maturin

**Option A: Install uv (recommended, faster)**
```bash
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option B: Install maturin**
```bash
pip install maturin
# or
cargo install maturin
```

## Step 2: Build and Install

Navigate to the Python bindings directory:

```bash
cd /path/to/baygraph/crates/grafial-python
```

### Using uv (recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode (this builds the Rust extension)
uv pip install -e .
```

### Using maturin directly

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install maturin in the virtual environment
pip install maturin

# Build and install in development mode
maturin develop --release
```

## Step 3: Verify Installation

Test that the module can be imported:

```bash
python3 -c "import grafial; print('Grafial imported successfully!')"
```

If you see "Grafial imported successfully!" without errors, you're good to go!

## Step 4: Run Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_basic.py -v
```

## Step 5: Try It Out

Create a test script `test_example.py`:

```python
import grafial

# Simple Grafial program
source = """
schema Test {
    node Person { }
    edge KNOWS { }
}

belief_model TestBeliefs on Test {
    edge KNOWS { exist ~ BernoulliPosterior(prior=0.5, pseudo_count=2.0) }
}
"""

# Compile the program
program = grafial.compile(source)

# Check what we got
print(f"Schemas: {program.get_schema_names()}")
print(f"Models: {program.get_belief_model_names()}")
print(f"Program: {program}")
```

Run it:

```bash
python3 test_example.py
```

You should see output like:
```
Schemas: ['Test']
Models: ['TestBeliefs']
Program: Program(schemas=1, models=1, evidences=0, rules=0, flows=0)
```

## Common Workflows

### Rebuilding After Code Changes

If you modify the Rust code in `src/lib.rs`, rebuild:

```bash
# With uv
uv pip install -e . --force-reinstall

# With maturin
maturin develop --release
```

### Development vs Release Builds

For faster iteration during development:
```bash
maturin develop  # Debug build (faster compilation, slower runtime)
```

For production-like performance:
```bash
maturin develop --release  # Release build (slower compilation, faster runtime)
```

### Using in Another Python Project

**Option 1: Install from local path**

```bash
# In your other project's virtual environment
pip install /path/to/baygraph/crates/grafial-python
```

**Option 2: Editable install (for active development)**

```bash
# In your other project's virtual environment
pip install -e /path/to/baygraph/crates/grafial-python
```

Then when you change the Rust code, rebuild:
```bash
cd /path/to/baygraph/crates/grafial-python
maturin develop --release
```

## Building for Distribution

### Build Wheels

To create distributable wheel files:

```bash
cd crates/grafial-python
maturin build --release
```

This creates `.whl` files in the `dist/` directory that can be:
- Shared with others
- Uploaded to PyPI
- Installed with `pip install dist/grafial-*.whl`

### Build Source Distribution

```bash
maturin sdist
```

This creates a source distribution that can be built on any platform.

## Troubleshooting

### "maturin: command not found"

Install maturin:
```bash
pip install maturin
# or
cargo install maturin
```

### "Can't find Python"

Set the Python path explicitly:
```bash
export PYO3_PYTHON=$(which python3)
maturin develop --release
```

Or specify Python version:
```bash
maturin develop --release --python python3.11
```

### "ModuleNotFoundError: No module named 'grafial'"

- Ensure you've activated the virtual environment: `source .venv/bin/activate`
- Reinstall: `uv pip install -e .` or `maturin develop --release`
- Check installation: `pip list | grep grafial`

### Build Errors

- Ensure Rust is up to date: `rustup update`
- Clean and rebuild: `cargo clean && maturin develop --release`
- Check Python version: `python3 --version` (needs 3.8+)

## Next Steps

- Read `README.md` for detailed documentation
- Check `tests/` directory for more examples
- See `../grafial-examples/` for Grafial program examples

