# Language Server Protocol (LSP) Implementation Plan

This document outlines how to add advanced IDE features beyond syntax highlighting using the Language Server Protocol.

## Current State

- ✅ **Syntax Highlighting**: TextMate grammar provides basic highlighting
- ❌ **LSP Features**: Not yet implemented

## LSP Features to Implement

### Phase 1: Diagnostics (Error Reporting) - **Highest Priority**

**What it does:**
- Shows parse errors with red squiggles
- Shows validation errors (e.g., "prob() used on non-edge variable")
- Real-time error checking as you type

**Implementation:**
- Create LSP server in Rust (new crate: `grafial-lsp`)
- Use existing `parse_and_validate()` from `grafial` crate
- Convert `ExecError` to LSP `Diagnostic` with line/column positions
- Publish diagnostics on file open, save, and change

**Effort:** 2-3 days

**Dependencies:**
- `tower-lsp` or `lsp-server` crate
- Pest parser position information (need to track line/column)

---

### Phase 2: Hover Documentation - **High Value**

**What it does:**
- Hover over `prob()` → shows "Returns the mean probability of edge existence"
- Hover over `GaussianPosterior` → shows parameter documentation
- Hover over schema names → shows definition location

**Implementation:**
- Implement `textDocument/hover` LSP method
- Build documentation map from AST
- Return formatted markdown with examples

**Effort:** 1-2 days

---

### Phase 3: Code Completion (IntelliSense) - **High Value**

**What it does:**
- Type `pro` → suggests `prob()`, `prob_vector()`
- Type `Gaus` → suggests `GaussianPosterior`
- Type `schema ` → suggests existing schema names
- Context-aware: after `prob(` → suggests edge variables from current rule

**Implementation:**
- Implement `textDocument/completion` LSP method
- Build completion lists from:
  - Keywords
  - Built-in functions
  - Schema/belief model/rule/flow names (from current file + imports)
  - Pattern variables (context-aware in rules)
  - Posterior types and parameters

**Effort:** 3-4 days

---

### Phase 4: Go to Definition - **Medium Priority**

**What it does:**
- `Cmd+Click` on `Social` in `belief_model M on Social` → jumps to schema definition
- `Cmd+Click` on `TransferAndDisconnect` → jumps to rule definition
- Works across files (if we add import support)

**Implementation:**
- Implement `textDocument/definition` LSP method
- Build symbol index from AST
- Track definition locations (line/column)
- Support workspace-wide symbol lookup

**Effort:** 2-3 days

---

### Phase 5: Document Symbols (Outline View) - **Medium Priority**

**What it does:**
- Shows tree view of schemas, rules, flows in sidebar
- Quick navigation to declarations
- Symbol search (`Cmd+Shift+O`)

**Implementation:**
- Implement `textDocument/documentSymbol` LSP method
- Build hierarchical symbol tree from AST
- Include symbols: schemas, nodes, edges, belief models, rules, flows, metrics

**Effort:** 1-2 days

---

### Phase 6: Code Actions (Quick Fixes) - **Nice to Have**

**What it does:**
- Error: `prob(A)` where `A` is a node → Quick fix: "Did you mean `E[A.attr]`?"
- Error: Unknown schema → Quick fix: "Create schema 'X'"
- Auto-format on save

**Implementation:**
- Implement `textDocument/codeAction` LSP method
- Analyze diagnostics to suggest fixes
- Optional: Implement formatter

**Effort:** 3-5 days

---

## Implementation Architecture

### Option A: Rust LSP Server (Recommended)

**Structure:**
```
grafial-lsp/
├── Cargo.toml
├── src/
│   ├── main.rs          # LSP server entry point
│   ├── server.rs        # LSP protocol handlers
│   ├── diagnostics.rs  # Error reporting
│   ├── hover.rs         # Documentation on hover
│   ├── completion.rs    # Code completion
│   └── symbols.rs       # Symbol indexing
└── README.md
```

**Dependencies:**
- `tower-lsp` or `lsp-server` (LSP protocol)
- `grafial` (parser/validator)
- `serde` (JSON serialization)

**Pros:**
- Reuses existing Rust parser/validator
- Fast and efficient
- Type-safe

**Cons:**
- Requires Rust toolchain
- Need to track source positions (line/column) in parser

---

### Option B: TypeScript/Node.js LSP Server

**Structure:**
```
grafial-lsp/
├── package.json
├── src/
│   ├── server.ts        # LSP server
│   └── handlers/        # Feature implementations
└── tsconfig.json
```

**Dependencies:**
- `vscode-languageserver` (Node.js LSP library)
- Call Rust CLI tool for parsing (or port parser to JS)

**Pros:**
- Easier integration with VS Code extension
- Can use existing VS Code APIs

**Cons:**
- Need to call external process for parsing (slower)
- Or duplicate parser logic in TypeScript

---

## Recommended Approach

**Start with Option A (Rust LSP Server):**

1. **Create `grafial-lsp` crate** alongside main `grafial` crate
2. **Enhance parser** to track line/column positions (Pest provides this)
3. **Implement Phase 1 (Diagnostics)** first - highest value
4. **Add to extension** by configuring LSP client in `package.json`

**Extension Integration:**

Update `grafial-vscode/package.json`:
```json
{
  "contributes": {
    "languages": [...],
    "grammars": [...],
    "configuration": {
      "type": "object",
      "title": "Grafial",
      "properties": {
        "grafial.lsp.enable": {
          "type": "boolean",
          "default": true,
          "description": "Enable LSP features"
        }
      }
    }
  },
  "activationEvents": ["onLanguage:grafial"]
}
```

Create `grafial-vscode/src/extension.ts`:
```typescript
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

export function activate(context: vscode.ExtensionContext) {
    const serverOptions: ServerOptions = {
        command: 'grafial-lsp',  // Path to LSP server binary
        args: []
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'grafial' }]
    };

    const client = new LanguageClient('grafial', 'Grafial Language Server', serverOptions, clientOptions);
    client.start();
}
```

---

## Priority Order

1. **Phase 1: Diagnostics** (2-3 days) - Essential for catching errors
2. **Phase 2: Hover** (1-2 days) - High user value, relatively easy
3. **Phase 3: Completion** (3-4 days) - Major productivity boost
4. **Phase 4: Go to Definition** (2-3 days) - Navigation improvement
5. **Phase 5: Document Symbols** (1-2 days) - Nice navigation feature
6. **Phase 6: Code Actions** (3-5 days) - Advanced feature

**Total estimated effort:** 12-19 days for full LSP implementation

---

## Next Steps

1. Enhance syntax highlighting (improve granularity) ✅ (in progress)
2. Create `grafial-lsp` crate structure
3. Implement Phase 1 (Diagnostics) as proof of concept
4. Integrate LSP server into VS Code extension
5. Iterate on remaining features

