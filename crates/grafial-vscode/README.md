# Grafial Language Support for VS Code / Cursor

Syntax highlighting and Language Server (LSP) features for the Grafial Bayesian Belief Graph Language.

## Features

- **Syntax Highlighting**: Full support for Grafial syntax including:
  - Keywords (`schema`, `belief_model`, `rule`, `flow`, etc.)
  - Posterior types (`Gaussian`, `Bernoulli`, `Categorical`; legacy aliases also highlighted)
  - Evidence modes (`present`, `absent`, `chosen`, etc.)
  - Built-in functions (`prob`, `prob_correlated`, `credible`, `degree`, `E[]`, `winner`, `entropy`, etc.)
  - Pattern matching syntax
  - Expressions and operators

- **Language Server (Phase 1)**:
  - Real-time diagnostics (syntax and validation errors)
  - Red squiggles with parse error locations (via Pest → LSP range)

- **Editor Features**:
  - Comment toggling (`//` and `/* */`)
  - Bracket matching
  - Auto-closing pairs
  - Smart indentation

## Installation

### From Source (Syntax Only)

1. Clone or download this repository
2. Open VS Code / Cursor
3. Press `F5` to open Extension Development Host
4. Open a `.grafial` file to see syntax highlighting

### Enable LSP (Development)

The LSP server is a Rust binary (`grafial-lsp`). Build it, then point the extension to the binary path:

```bash
# in repo root
cargo build -p grafial-lsp --release

# server binary location
# macOS/Linux: target/release/grafial-lsp
# Windows:     target\\release\\grafial-lsp.exe
```

Server resolution order:
- `grafial.serverPath` (if set)
- local workspace build (`target/release/grafial-lsp`)
- `grafial-lsp` on `PATH`

Set `grafial.serverPath` only when you want to override that default behavior.

Note: This extension uses `vscode-languageclient`. For packaging, install dev dependencies and package as usual (requires internet to fetch npm packages):

```bash
cd crates/grafial-vscode
npm install
vsce package
```

### Package and Install

To create a `.vsix` package:

```bash
npm install -g vsce
cd crates/grafial-vscode
vsce package
```

Then install the `.vsix` file:

**VS Code:**
- Command Palette (`Cmd+Shift+P`): `Extensions: Install from VSIX...`
- Or drag `.vsix` file into VS Code window

**Cursor:**
- **Method 1**: Open Extensions view (`Cmd+Shift+X`), click `...` menu (top right), select "Install from VSIX..."
- **Method 2**: Command Palette (`Cmd+Shift+P`) → type "Install from VSIX" and select it
- **Method 3** (Manual): Copy the `grafial-vscode` folder to:
  - macOS/Linux: `~/.cursor/extensions/grafial-0.1.0/`
  - Windows: `%USERPROFILE%\.cursor\extensions\grafial-0.1.0\`
  Then restart Cursor

## Compatibility

- **VS Code**: 1.74.0 or later
- **Cursor**: All versions (Cursor is based on VS Code)

## Language Features

The extension recognizes:
- File extension: `.grafial`
- Language ID: `grafial`

## Contributing

Contributions welcome! The TextMate grammar is in `syntaxes/grafial.tmLanguage.json`. The LSP server lives in `crates/grafial-lsp` and uses `tower-lsp`.
