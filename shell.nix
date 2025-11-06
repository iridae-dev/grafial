{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Core developer toolchain for Grafial
  packages = with pkgs; [
    # LLVM coverage tools (the main thing we need for coverage)
    llvmPackages.llvm        # provides llvm-cov, llvm-profdata
    llvmPackages.bintools

    # Python + PyO3 build tool
    python312
    # maturin might not be available in all nixpkgs channels; install via pip if needed
    # python312Packages.maturin
    python312Packages.setuptools
    python312Packages.wheel
    python312Packages.pip

    # Native build helpers and common C libs
    pkg-config
    openssl
  ];

  # Helpful defaults
  RUST_BACKTRACE = "1";
  RUST_LOG = "info";
  CARGO_TERM_COLOR = "always";

  # Ensure maturin/pyo3 uses the nix-provided Python
  # (maturin will detect this automatically; this is explicit and robust)
  PYO3_PYTHON = "${pkgs.python312}/bin/python3";

  shellHook = ''
    echo "Grafial dev shell loaded"
    echo "- Using system Rust toolchain (if available)"
    echo "- rustc:       $(rustc --version 2>/dev/null || echo 'not found - install via rustup')"
    echo "- cargo:       $(cargo --version 2>/dev/null || echo 'not found - install via rustup')"
    echo "- python:      $(python3 --version)"
    echo "- maturin:     $(maturin --version || echo 'not installed (install via: pip install maturin)')"
    echo "- llvm-cov:    $(llvm-cov --version || true)"
    echo "- profdata:    $(llvm-profdata merge --help 2>&1 | head -1 || echo 'available')"
    echo "- cargo-llvm-cov: $(cargo llvm-cov --version 2>/dev/null || echo 'not installed (install via: cargo install cargo-llvm-cov)')"
  '';
}
