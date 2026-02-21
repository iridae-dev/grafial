// Minimal VSCode extension entry to start the Grafial LSP server.
// Supports either:
// - `grafial.serverPath` as a command name on PATH (e.g. grafial-lsp)
// - `grafial.serverPath` as an absolute/relative executable path

const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
const path = require('path');
const os = require('os');
const fs = require('fs');

function workspaceRoot() {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
}

function defaultCommandName() {
  return process.platform === 'win32' ? 'grafial-lsp.exe' : 'grafial-lsp';
}

function isPathLike(candidate) {
  return (
    path.isAbsolute(candidate) ||
    candidate.startsWith('~') ||
    candidate.startsWith('.') ||
    candidate.includes('/') ||
    candidate.includes('\\')
  );
}

function resolveServerCommand(configuredPath) {
  if (!configuredPath || configuredPath.trim() === '') {
    return null;
  }

  let candidate = configuredPath.trim();

  if (!isPathLike(candidate)) {
    // Bare command name (e.g. "grafial-lsp"): resolve via PATH.
    return candidate;
  }

  if (candidate.startsWith('~') && !path.isAbsolute(candidate)) {
    candidate = path.join(os.homedir(), candidate.slice(1));
  }

  if (!path.isAbsolute(candidate)) {
    candidate = path.resolve(workspaceRoot(), candidate);
  }

  return candidate;
}

function validateExecutablePath(serverPath) {
  if (!fs.existsSync(serverPath)) {
    return `Server binary not found at ${serverPath}`;
  }

  if (process.platform === 'win32') {
    return null;
  }

  try {
    fs.accessSync(serverPath, fs.constants.X_OK);
  } catch (err) {
    return `Server binary is not executable: ${serverPath}`;
  }

  return null;
}

function maybeWorkspaceBuild() {
  const candidate = path.join(workspaceRoot(), 'target', 'release', defaultCommandName());
  return validateExecutablePath(candidate) ? null : candidate;
}

/** @type {LanguageClient | null} */
let client = null;

function activate(context) {
  console.log('[Grafial] Extension activating...');
  
  try {
    const config = vscode.workspace.getConfiguration('grafial');
    const configuredValue = config.get('serverPath');
    const configuredCommand = resolveServerCommand(configuredValue);
    const workspaceBuild = maybeWorkspaceBuild();

    let serverCommand = configuredCommand || workspaceBuild || defaultCommandName();
    let commandSource = configuredCommand
      ? 'grafial.serverPath'
      : workspaceBuild
      ? 'workspace target/release build'
      : 'PATH';

    // If config is left at default command name and a local build exists, prefer local build.
    if (configuredCommand === defaultCommandName() && workspaceBuild) {
      serverCommand = workspaceBuild;
      commandSource = 'workspace target/release build';
    }

    if (path.isAbsolute(serverCommand)) {
      const validationError = validateExecutablePath(serverCommand);
      if (validationError) {
        const errorMsg = `[Grafial] ERROR: ${validationError}`;
        console.error(errorMsg);
        vscode.window.showErrorMessage(errorMsg);
        return;
      }
    }

    console.log('[Grafial] Server command:', serverCommand, `(source: ${commandSource})`);

    // Create output channel first
    const outputChannel = vscode.window.createOutputChannel('Grafial Language Server');
    outputChannel.appendLine('[Grafial] Extension activating...');
    outputChannel.appendLine(`[Grafial] Server command: ${serverCommand}`);
    outputChannel.appendLine(`[Grafial] Command source: ${commandSource}`);

    const serverOptions = {
      command: serverCommand,
      args: [],
      options: { 
        cwd: workspaceRoot(),
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
      console.error(errorMsg);
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
