#!/bin/bash
# Setup script for code coverage tools

echo "Setting up code coverage for Baygraph..."
echo ""

# Install llvm-tools-preview
echo "1. Installing llvm-tools-preview..."
rustup component add llvm-tools-preview

if [ $? -eq 0 ]; then
    echo "✓ llvm-tools-preview installed successfully"
else
    echo "✗ Failed to install llvm-tools-preview"
    exit 1
fi

echo ""
echo "2. Checking cargo-llvm-cov installation..."
if ! command -v cargo-llvm-cov &> /dev/null; then
    echo "Installing cargo-llvm-cov..."
    cargo install cargo-llvm-cov
    if [ $? -eq 0 ]; then
        echo "✓ cargo-llvm-cov installed successfully"
    else
        echo "✗ Failed to install cargo-llvm-cov"
        exit 1
    fi
else
    echo "✓ cargo-llvm-cov already installed"
fi

echo ""
echo "Setup complete! You can now run:"
echo "  cargo llvm-cov --html --open"
echo ""
