# Quick Test Guide for Grafial LSP in Cursor

Replace `<repo-root>` with the absolute path to your local `baygraph` checkout.

## Method 1: Extension Development Host (Easiest - No Settings Needed)

1. **Open the extension folder in Cursor:**
   ```bash
   cd <repo-root>/crates/grafial-vscode
   cursor .
   ```

2. **Press F5** (or Cmd+F5 on Mac) to launch Extension Development Host

3. **In the new window:**
   - Open any `.grafial` file (e.g., `crates/grafial-examples/minimal.grafial`)
   - Hover over `prob`, `E`, `Gaussian`, `Bernoulli`, etc.
   - You should see hover tooltips!

   By default, the extension will try local `target/release/grafial-lsp` first and then `grafial-lsp` on `PATH`.

## Method 2: Set Server Path via Settings JSON

If you want to configure it manually in Cursor:

1. **Open Command Palette** (Cmd+Shift+P)
2. Type: `Preferences: Open User Settings (JSON)`
3. Add this line:
   ```json
   {
     "grafial.serverPath": "<repo-root>/target/release/grafial-lsp"
   }
   ```

## Method 3: Edit Settings via UI

1. **Open Settings** (Cmd+,)
2. **Search for:** `grafial`
3. If it doesn't appear, the extension might not be installed. Use Method 1 instead.

## Method 4: Test Server Directly (Verify Binary Works)

Test that the server binary responds:

```bash
# Quick test - should show JSON response
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | <repo-root>/target/release/grafial-lsp
```

If you see JSON output, the server is working.

## Troubleshooting

**If the setting doesn't appear:**
- The extension needs to be installed/loaded first
- Use Method 1 (F5) to test without installing
- Or check Cursor's Output panel: View → Output → Select "Grafial Language Server"

**If hover doesn't work:**
- Check Cursor's Output panel for errors
- Verify the binary exists: `ls -lh <repo-root>/target/release/grafial-lsp`
- Make sure you're hovering over valid tokens (not whitespace)
