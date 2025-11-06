# Grafial Language Support for VS Code / Cursor

Syntax highlighting for the Grafial Bayesian Belief Graph Language.

## Features

- **Syntax Highlighting**: Full support for Grafial syntax including:
  - Keywords (`schema`, `belief_model`, `rule`, `flow`, etc.)
  - Posterior types (`GaussianPosterior`, `BernoulliPosterior`, `CategoricalPosterior`)
  - Evidence modes (`present`, `absent`, `chosen`, etc.)
  - Built-in functions (`prob`, `degree`, `E[]`, `winner`, `entropy`, etc.)
  - Pattern matching syntax
  - Expressions and operators

- **Editor Features**:
  - Comment toggling (`//` and `/* */`)
  - Bracket matching
  - Auto-closing pairs
  - Smart indentation

## Installation

### From Source

1. Clone or download this repository
2. Open VS Code / Cursor
3. Press `F5` to open Extension Development Host
4. Open a `.grafial` file to see syntax highlighting

### Package and Install

To create a `.vsix` package:

```bash
npm install -g vsce
cd grafial-vscode
vsce package
```

Then install the `.vsix` file:

**VS Code:**
- Command Palette (`Cmd+Shift+P`): `Extensions: Install from VSIX...`
- Or drag `.vsix` file into VS Code window

**Cursor:**
- **Method 1**: Open Extensions view (`Cmd+Shift+X`), click `...` menu (top right), select "Install from VSIX..."
- **Method 2**: Command Palette (`Cmd+Shift+P`) → type "Install from VSIX" and select it
- **Method 3** (Easiest - macOS/Linux): Run the install script:
  ```bash
  cd grafial-vscode
  ./install-cursor.sh
  ```
- **Method 4** (Manual): Copy the `grafial-vscode` folder to:
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

Improvements to syntax highlighting are welcome! The grammar is defined in `syntaxes/grafial.tmLanguage.json` using TextMate grammar format.

