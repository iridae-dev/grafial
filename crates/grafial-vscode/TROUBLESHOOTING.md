# Troubleshooting "Activating..." Issue

Replace `<repo-root>` with the absolute path to your local `baygraph` checkout.

If the extension is stuck in "Activating..." state:

## Step 1: Check Developer Console

1. **Command Palette** (`Cmd+Shift+P`)
2. **Type**: `Developer: Toggle Developer Tools`
3. **Click Console tab**
4. Look for `[Grafial]` log messages or red errors
5. **Share any errors you see**

## Step 2: Check Output Channel

1. **View → Output** (or `Cmd+Shift+U`)
2. **Dropdown → "Grafial Language Server"**
3. Look for error messages or where it stops

## Step 3: Test Server Binary Manually

```bash
# Test if server starts (should hang waiting for input, that's normal)
<repo-root>/target/release/grafial-lsp
```

Press `Ctrl+C` to stop it. If it crashes immediately, that's the problem.

## Step 4: Check Extension Logs

1. **Command Palette** (`Cmd+Shift+P`)
2. **Type**: `Developer: Show Running Extensions`
3. Find "Grafial Language Support"
4. Click it to see details/errors

## Step 5: Reinstall Extension

1. **Extensions** (`Cmd+Shift+X`)
2. Find "Grafial Language Support"
3. **Uninstall** it
4. **Install from VSIX** again:
   - Click `...` menu → "Install from VSIX..."
   - Select: `<repo-root>/crates/grafial-vscode/grafial-0.1.0.vsix`
5. **Reload Window** (`Cmd+Shift+P` → "Developer: Reload Window")

## Common Issues

**Binary not found:**
- Check: `ls -lh <repo-root>/target/release/grafial-lsp`
- Rebuild: `cargo build -p grafial-lsp --release`

**Binary not executable:**
- Fix: `chmod +x <repo-root>/target/release/grafial-lsp`

**Server crashes on start:**
- Check Developer Console for error messages
- The server might have a bug - check the LSP server code

**Extension hangs:**
- Usually means `client.start()` is waiting for the server
- Server might be blocked or waiting for input incorrectly
