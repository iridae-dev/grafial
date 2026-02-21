# Testing Grafial LSP in Cursor - Step by Step

Replace `<repo-root>` with the absolute path to your local `baygraph` checkout.

## Step 1: Reinstall Updated Extension

The VSIX has been rebuilt with the updated launcher behavior. Reinstall it:

1. **Open Extensions view** in Cursor (`Cmd+Shift+X`)
2. **Click the "..." menu** (top right)
3. **Select "Install from VSIX..."**
4. **Choose**: `<repo-root>/crates/grafial-vscode/grafial-0.1.0.vsix`
5. **Reload Cursor** when prompted (or `Cmd+Shift+P` → "Developer: Reload Window")

## Step 2: Verify Extension is Active

1. **Open a `.grafial` file** (e.g., `crates/grafial-examples/minimal.grafial`)
2. Check bottom-right corner - should show "Grafial" as the language
3. **Command Palette** (`Cmd+Shift+P`) → "Developer: Show Running Extensions"
4. Look for "Grafial Language Support" - should show as activated

## Step 3: Check LSP Server is Running

1. **View → Output** (or `Cmd+Shift+U`)
2. **Select dropdown** - look for "Grafial Language Server" or "Log (Window)"
3. You should see connection messages (or errors if something's wrong)

## Step 4: Test Hover

1. In your `.grafial` file, hover over:
   - `prob` - should show hover docs
   - `E` - should show hover docs  
   - `Gaussian` - should show hover docs
   - Any schema/model/rule/flow name from your file

## Step 5: If It's Not Working

### Check Server Path (if needed)

`grafial.serverPath` is optional. Set it only if you want to force a specific binary:

1. **Command Palette** (`Cmd+Shift+P`)
2. **Type**: `Preferences: Open User Settings (JSON)`
3. **Add**:
   ```json
   {
     "grafial.serverPath": "<repo-root>/target/release/grafial-lsp"
   }
   ```

### Check for Errors

1. **Command Palette** (`Cmd+Shift+P`)
2. **Type**: `Developer: Toggle Developer Tools`
3. **Check Console tab** for JavaScript errors
4. **Check Output panel** for server errors

### Verify Binary Works

```bash
# Test server manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | <repo-root>/target/release/grafial-lsp
```

Should see JSON response. If not, there's a binary issue.

## What Should Work

- ✅ Syntax highlighting (already working)
- ✅ Hover over built-ins (`prob`, `E`, `winner`, `entropy`, `degree`)
- ✅ Hover over posterior types (`Gaussian`, `Bernoulli`, `Categorical`; legacy aliases also supported)
- ✅ Hover over schema/model/rule/flow names
- ✅ Error diagnostics (red squiggles on syntax errors)

## Next Steps After Verification

Once hover is working, you can test:
- Open a file with errors - should see red squiggles
- Hover over different tokens
- Check the Output panel to see LSP messages
