# Setting Up Grafial LSP in Cursor

Replace `<repo-root>` with the absolute path to your local `baygraph` checkout.

## Step 1: Build the LSP Server

Expected local build path:
```
<repo-root>/target/release/grafial-lsp
```

If you need to rebuild:
```bash
cd <repo-root>
cargo build -p grafial-lsp --release
```

## Step 2: Install Extension Dependencies

If you haven't already, install the extension dependencies:

```bash
cd <repo-root>/crates/grafial-vscode
npm install
```

This installs `vscode-languageclient` and related dependencies.

## Step 3: Configure Cursor to Use the LSP Server

You have two options:

### Option A: Add Binary to PATH (Recommended)

Add the binary to your PATH so Cursor can find it:

**macOS/Linux:**
```bash
# Add to your ~/.zshrc or ~/.bashrc
export PATH="<repo-root>/target/release:$PATH"
```

Then restart Cursor.

### Option B: Configure Extension Setting

1. Open Cursor Settings (Cmd+,)
2. Search for "grafial"
3. Find "Grafial: Server Path"
4. Set it to the absolute path:
   ```
   <repo-root>/target/release/grafial-lsp
   ```

Or add to your Cursor settings JSON:
```json
{
  "grafial.serverPath": "<repo-root>/target/release/grafial-lsp"
}
```

If you leave `grafial.serverPath` unset, the extension will try:
1. local workspace build (`target/release/grafial-lsp`)
2. `grafial-lsp` on `PATH`

## Step 4: Load the Extension in Cursor

Since you're developing the extension, you have a few options:

### Option A: Package and Install (For Testing)

```bash
cd <repo-root>/crates/grafial-vscode
npm install -g vsce  # If you don't have vsce installed
vsce package
```

Then in Cursor:
1. Open Extensions view (Cmd+Shift+X)
2. Click the "..." menu (top right)
3. Select "Install from VSIX..."
4. Choose the generated `.vsix` file

### Option B: Development Mode (Recommended for Testing)

1. Open the `crates/grafial-vscode` folder in Cursor
2. Press `F5` (or Cmd+F5) to launch Extension Development Host
3. In the new window, open a `.grafial` file
4. The LSP server should automatically start

### Option C: Manual Installation

Copy the extension folder to Cursor's extensions directory:

```bash
# macOS
cp -r <repo-root>/crates/grafial-vscode ~/.cursor/extensions/grafial-0.1.0/

# Then install dependencies
cd ~/.cursor/extensions/grafial-0.1.0
npm install
```

Restart Cursor.

## Step 5: Verify It's Working

1. Open a `.grafial` file in Cursor
2. Hover over:
   - Built-in functions: `prob`, `E`, `winner`, `entropy`, `degree`
   - Posterior types: `Gaussian`, `Bernoulli`, `Categorical` (legacy aliases also supported)
   - Schema/model/rule/flow names defined in your file
3. You should see hover tooltips with documentation

## Troubleshooting

### Check LSP Server is Running

Open Cursor's Output panel (View → Output), select "Grafial Language Server" from the dropdown. You should see logs.

### Check Extension is Loaded

1. Open Command Palette (Cmd+Shift+P)
2. Type "Developer: Show Running Extensions"
3. Look for "Grafial Language Support"

### Manual Server Test

Test the server directly:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | <repo-root>/target/release/grafial-lsp
```

You should see a JSON response (the server will exit after receiving EOF).

### Check Server Path Configuration

In Cursor, open Settings and verify `grafial.serverPath` points to the correct binary path.
