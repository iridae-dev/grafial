// Minimal VSCode extension entry to start the Grafial LSP server.
// Expects a `grafial-lsp` binary on PATH or a configured absolute path.

const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const path = require('path');
const os = require('os');
const fs = require('fs');

function resolveServerPath(configuredPath) {
  if (!configuredPath || configuredPath.trim() === '') {
    return null;
  }

  let candidate = configuredPath.trim();

  if (candidate.startsWith('~')) {
    candidate = path.join(os.homedir(), candidate.slice(1));
  }

  if (!path.isAbsolute(candidate)) {
    const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    candidate = workspacePath ? path.resolve(workspacePath, candidate) : path.resolve(candidate);
  }

  return candidate;
}

/** @type {LanguageClient | null} */
let client = null;

function activate(context) {
  console.log('[Grafial] Extension activating...');
  
  try {
    const config = vscode.workspace.getConfiguration('grafial');
    const defaultPath = '/Users/charleshinshaw/Desktop/baygraph/target/release/grafial-lsp';
    const configuredPath = config.get('serverPath');
    const serverPath = resolveServerPath(configuredPath) || defaultPath;
    
    console.log('[Grafial] Server path:', serverPath);
    
    // Check if binary exists
    if (!fs.existsSync(serverPath)) {
      const errorMsg = `[Grafial] ERROR: Server binary not found at ${serverPath}`;
      console.error(errorMsg);
      vscode.window.showErrorMessage(errorMsg);
      return;
    }
    
    // Check if binary is executable
    try {
      fs.accessSync(serverPath, fs.constants.X_OK);
    } catch (err) {
      const errorMsg = `[Grafial] ERROR: Server binary is not executable: ${serverPath}`;
      console.error(errorMsg);
      vscode.window.showErrorMessage(errorMsg);
      return;
    }

    // Create output channel first
    const outputChannel = vscode.window.createOutputChannel('Grafial Language Server');
    outputChannel.appendLine('[Grafial] Extension activating...');
    outputChannel.appendLine(`[Grafial] Server path: ${serverPath}`);

    const serverOptions = {
      command: serverPath,
      args: [],
      options: { 
        cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd(),
        env: process.env
      },
      transport: TransportKind.stdio,
    };

    const clientOptions = {
      documentSelector: [{ scheme: 'file', language: 'grafial' }],
      outputChannel: outputChannel,
    };

    console.log('[Grafial] Creating LanguageClient...');
    outputChannel.appendLine('[Grafial] Creating LanguageClient...');
    
    client = new LanguageClient(
      'grafialLsp', 
      'Grafial Language Server', 
      serverOptions, 
      clientOptions
    );
    
    console.log('[Grafial] Starting client...');
    outputChannel.appendLine('[Grafial] Starting client...');
    
    client.start().then(() => {
      console.log('[Grafial] Client started successfully');
      outputChannel.appendLine('[Grafial] Client started successfully');
    }).catch(err => {
      const errorMsg = `[Grafial] Failed to start client: ${err.message}`;
      console.error(errorMsg, err);
      outputChannel.appendLine(`ERROR: ${errorMsg}`);
      outputChannel.appendLine(err.stack || err.toString());
      outputChannel.show(true);
      vscode.window.showErrorMessage(errorMsg);
    });
    
    context.subscriptions.push(client);
    console.log('[Grafial] Extension activation complete');
    
  } catch (err) {
    console.error('[Grafial] Activation error:', err);
    vscode.window.showErrorMessage(`[Grafial] Activation failed: ${err.message}`);
  }
}

function deactivate() {
  console.log('[Grafial] Extension deactivating...');
  if (!client) return undefined;
  return client.stop();
}

module.exports = { activate, deactivate };

