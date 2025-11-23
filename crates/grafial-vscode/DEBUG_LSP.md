# Debugging Grafial LSP in Cursor

## Step 1: Verify Extension is Loaded

1. **Open Command Palette** (`Cmd+Shift+P`)
2. Type: `Developer: Show Running Extensions`
3. Look for "Grafial Language Support" or "grafial"
4. Check if it shows as "Activated" or has any errors

## Step 2: Check Extension Output Logs

1. **Open Output Panel**: View → Output (or `Cmd+Shift+U`)
2. **Select dropdown**: Choose "Grafial Language Server" or "Log (Window)"
3. Look for:
   - Connection messages
   - Error messages about server path
   - Any crash/error logs

## Step 3: Verify Server Binary Path

The extension is configured to use:
```
/Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
```

Check if this file exists:
```bash
ls -lh /Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
```

If it doesn't exist, rebuild:
```bash
cd /Users/charleshinshaw/Desktop/baygraph
cargo build -p grafial-lsp --release
```

## Step 4: Test Server Manually

Test that the server binary works:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | /Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
```

Should see JSON response. If not, the binary might have issues.

## Step 5: Set Server Path in Settings

Since the extension is installed, you can set it in settings:

1. **Command Palette** (`Cmd+Shift+P`)
2. Type: `Preferences: Open User Settings (JSON)`
3. Add:
   ```json
   {
     "grafial.serverPath": "/Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp"
   }
   ```
4. **Reload Cursor** (or reload window: `Cmd+Shift+P` → "Developer: Reload Window")

## Step 6: Check for Extension Errors

1. **Command Palette** (`Cmd+Shift+P`)
2. Type: `Developer: Toggle Developer Tools`
3. Check Console tab for any JavaScript errors

## Common Issues

**Extension not activating:**
- Open a `.grafial` file (extension activates on `onLanguage:grafial`)
- Check if file is recognized (should show "Grafial" in bottom-right language indicator)

**Server not starting:**
- Check Output panel for "Grafial Language Server"
- Verify binary path is correct and executable
- Try absolute path in settings

**Hover not working:**
- Make sure you're hovering over actual tokens (not whitespace)
- Check if diagnostics are working (errors should show red squiggles)
- Verify LSP server is connected (check Output panel)

