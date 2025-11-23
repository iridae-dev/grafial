# Setting Up Grafial LSP in Cursor

## Step 1: Build the LSP Server

The LSP server binary is already built at:
```
/Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
```

If you need to rebuild:
```bash
cd /Users/charleshinshaw/Desktop/baygraph
cargo build -p grafial-lsp --release
```

## Step 2: Install Extension Dependencies

If you haven't already, install the extension dependencies:

```bash
cd /Users/charleshinshaw/Desktop/baygraph/crates/grafial-vscode
npm install
```

This will install `vscode-languageclient` and other required dependencies.

## Step 3: Configure Cursor to Use the LSP Server

You have two options:

### Option A: Add Binary to PATH (Recommended)

Add the binary to your PATH so Cursor can find it:

**macOS/Linux:**
```bash
# Add to your ~/.zshrc or ~/.bashrc
export PATH="/Users/charleshinshaw/Desktop/baygraph/target/release:$PATH"
```

Then restart Cursor.

### Option B: Configure Extension Setting

1. Open Cursor Settings (Cmd+,)
2. Search for "grafial"
3. Find "Grafial: Server Path"
4. Set it to the absolute path:
   ```
   /Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
   ```

Or add to your Cursor settings JSON:
```json
{
  "grafial.serverPath": "/Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp"
}
```

## Step 4: Load the Extension in Cursor

Since you're developing the extension, you have a few options:

### Option A: Package and Install (For Testing)

```bash
cd /Users/charleshinshaw/Desktop/baygraph/crates/grafial-vscode
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
cp -r /Users/charleshinshaw/Desktop/baygraph/crates/grafial-vscode ~/.cursor/extensions/grafial-0.1.0/

# Then install dependencies
cd ~/.cursor/extensions/grafial-0.1.0
npm install
```

Restart Cursor.

## Step 5: Verify It's Working

1. Open a `.grafial` file in Cursor
2. Hover over:
   - Built-in functions: `prob`, `E`, `winner`, `entropy`, `degree`
   - Posterior types: `GaussianPosterior`, `BernoulliPosterior`, `CategoricalPosterior`
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
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | /Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp
```

You should see a JSON response (the server will exit after receiving EOF).

### Check Server Path Configuration

In Cursor, open Settings and verify `grafial.serverPath` points to the correct binary path.

