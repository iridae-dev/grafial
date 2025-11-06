{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Core developer toolchain for Grafial monorepo
  packages = with pkgs; [
    # Rust toolchain (install via rustup)
    # Note: Rust is not included here - use rustup for toolchain management
    
    # LLVM coverage tools
    llvmPackages.llvm        # provides llvm-cov, llvm-profdata
    llvmPackages.bintools

    # Python + PyO3 build tools
    python312
    python312Packages.setuptools
    python312Packages.wheel
    python312Packages.pip
    
    # Note: uv is not yet in stable nixpkgs - install via:
    # curl -LsSf https://astral.sh/uv/install.sh | sh
    # Or use: nix profile install nixpkgs#python312Packages.uv (if available)

    # Native build helpers and common C libs
    pkg-config
    openssl
  ];

  # Helpful defaults
  RUST_BACKTRACE = "1";
  RUST_LOG = "info";
  CARGO_TERM_COLOR = "always";

  # Ensure PyO3 uses the nix-provided Python
  # (maturin/uv will detect this automatically; this is explicit and robust)
  PYO3_PYTHON = "${pkgs.python312}/bin/python3";

  shellHook = ''
    echo "Grafial dev shell loaded"
    echo ""
    echo "Rust toolchain:"
    echo "  rustc:       $(rustc --version 2>/dev/null || echo 'not found - install via rustup')"
    echo "  cargo:       $(cargo --version 2>/dev/null || echo 'not found - install via rustup')"
    echo ""
    echo "Python toolchain:"
    echo "  python:      $(python3 --version)"
    echo "  uv:          $(uv --version 2>/dev/null || echo 'not installed (install via: curl -LsSf https://astral.sh/uv/install.sh | sh)')"
    echo "  maturin:     $(maturin --version 2>/dev/null || echo 'not installed (install via: pip install maturin or uv pip install maturin)')"
    echo ""
    echo "Coverage tools:"
    echo "  llvm-cov:    $(llvm-cov --version 2>/dev/null | head -1 || echo 'available')"
    echo "  profdata:    $(llvm-profdata merge --help 2>&1 | head -1 || echo 'available')"
    echo "  cargo-llvm-cov: $(cargo llvm-cov --version 2>/dev/null || echo 'not installed (install via: cargo install cargo-llvm-cov)')"
    echo ""
    echo "Monorepo structure:"
    echo "  crates/grafial-core/      - Core engine"
    echo "  crates/grafial-frontend/  - Parser, AST, validation"
    echo "  crates/grafial-cli/       - CLI tool"
    echo "  crates/grafial-python/    - Python bindings"
    echo "  crates/grafial-tests/     - Integration tests"
    echo ""
    echo "Quick start:"
    echo "  Build workspace:    cargo build --workspace --release"
    echo "  Run tests:          cargo test --workspace"
    echo "  Build CLI:          cargo build -p grafial-cli --release"
    echo "  Python bindings:    cd crates/grafial-python && uv venv && uv pip install -e ."
  '';
}
