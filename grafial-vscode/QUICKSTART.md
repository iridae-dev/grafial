# Quick Start Guide

## Testing in VS Code / Cursor

1. **Open the extension folder**:
   ```bash
   cd grafial-vscode
   code .  # or `cursor .` for Cursor
   ```

2. **Launch Extension Development Host**:
   - Press `F5` (or `Cmd+F5` on Mac)
   - This opens a new window with the extension loaded

3. **Test syntax highlighting**:
   - In the new window, open a `.grafial` file (e.g., `../examples/social.grafial`)
   - You should see syntax highlighting applied

4. **Make changes**:
   - Edit `syntaxes/grafial.tmLanguage.json` to adjust highlighting
   - Press `Ctrl+R` (or `Cmd+R` on Mac) in the Extension Development Host to reload

## Installing Locally (Development)

To install the extension in your main VS Code / Cursor instance:

1. **Package the extension**:
   ```bash
   npm install -g vsce
   vsce package
   ```
   This creates `grafial-0.1.0.vsix`

2. **Install the package**:

   **VS Code:**
   - `Cmd+Shift+P` → "Extensions: Install from VSIX..."
   - Select the `.vsix` file
   - Restart the editor

   **Cursor:**
   - **Method 1**: Open Extensions view (`Cmd+Shift+X`), click `...` menu (top right), select "Install from VSIX..."
   - **Method 2**: `Cmd+Shift+P` → type "Install from VSIX" and select it
   - **Method 3** (If VSIX doesn't work): Manually copy the extension folder:
     ```bash
     # macOS/Linux
     cp -r grafial-vscode ~/.cursor/extensions/grafial-0.1.0
     
     # Windows (PowerShell)
     Copy-Item -Recurse grafial-vscode "$env:USERPROFILE\.cursor\extensions\grafial-0.1.0"
     ```
     Then restart Cursor

3. **Verify**:
   - Open any `.grafial` file
   - Syntax highlighting should work automatically

## File Structure

```
grafial-vscode/
├── package.json              # Extension manifest
├── language-configuration.json  # Editor features (comments, brackets, etc.)
├── syntaxes/
│   └── grafial.tmLanguage.json  # TextMate grammar (syntax highlighting rules)
├── README.md                 # Extension documentation
└── QUICKSTART.md            # This file
```

## Customizing Syntax Highlighting

Edit `syntaxes/grafial.tmLanguage.json` to:
- Add new keywords
- Change color scopes
- Adjust pattern matching

The grammar uses TextMate syntax. See [TextMate Language Grammars](https://macromates.com/manual/en/language_grammars) for reference.

