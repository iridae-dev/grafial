# LLVM Optimization Guide for Grafial AST

## Overview

This document outlines strategies for using LLVM to optimize Grafial's AST execution. Grafial currently uses runtime interpretation of the AST, but LLVM can provide significant performance improvements through ahead-of-time (AOT) or just-in-time (JIT) compilation.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [LLVM Integration Strategy](#llvm-integration-strategy)
3. [Compilation Pipeline](#compilation-pipeline)
4. [Optimization Targets](#optimization-targets)
5. [Implementation Approach](#implementation-approach)
6. [Code Examples](#code-examples)
7. [Performance Considerations](#performance-considerations)
8. [References](#references)

---

## Current Architecture

### AST Structure

Grafial's `ProgramAst` (defined in `crates/grafial-frontend/src/ast.rs`) consists of:

```rust
pub struct ProgramAst {
    pub schemas: Vec<Schema>,           // Graph structure definitions
    pub belief_models: Vec<BeliefModel>, // Bayesian inference parameters
    pub evidences: Vec<EvidenceDef>,    // Observations
    pub rules: Vec<RuleDef>,            // Pattern-based transformations
    pub flows: Vec<FlowDef>,            // Execution pipelines
}
```

### Current Execution Model

**Runtime Interpretation**:
1. Parse Grafial source → AST
2. Validate AST semantics
3. Interpret AST directly in the execution engine
4. Pattern matching via graph queries
5. Expression evaluation via trait-based visitor pattern

**Performance Bottlenecks**:
- Dynamic dispatch for expression evaluation
- Repeated pattern matching for rules
- Interpretation overhead for tight loops
- Missed optimization opportunities (inlining, dead code elimination, constant folding)

---

## LLVM Integration Strategy

### Why LLVM?

LLVM provides:
- **Machine-independent IR**: Target multiple architectures
- **Rich optimization passes**: DCE, CSE, loop optimization, inlining, vectorization
- **JIT compilation**: Dynamic code generation via LLVM's ORC JIT
- **Mature ecosystem**: Rust bindings via `inkwell` crate

### Integration Levels

We can integrate LLVM at multiple levels:

1. **Expression JIT** (Quick Win): Compile hot expressions to native code
2. **Rule Compilation** (Medium): Compile pattern-matching and actions
3. **Flow AOT** (Advanced): Compile entire flows ahead-of-time
4. **Full Program Compilation** (Complete): Statically compile Grafial to executables

---

## Compilation Pipeline

### Proposed Architecture

```
Grafial Source
    ↓
[Pest Parser]
    ↓
ProgramAst (crates/grafial-frontend/src/ast.rs)
    ↓
[Semantic Validation]
    ↓
ProgramIR (crates/grafial-ir/src/lib.rs) ← Already exists!
    ↓
[NEW: LLVM IR Lowering]
    ↓
LLVM IR Module
    ↓
[LLVM Optimization Passes]
    ↓
┌─────────────────┬─────────────────┐
│   JIT Engine    │   AOT Compiler  │
│  (Development)  │   (Production)  │
└─────────────────┴─────────────────┘
         ↓                 ↓
  Native Functions    Standalone Binary
```

### Key Stages

1. **AST → IR**: Already designed in `grafial-ir` (needs population)
2. **IR → LLVM IR**: New lowering pass to generate LLVM IR
3. **LLVM Optimization**: Apply standard LLVM passes
4. **Code Generation**: JIT or AOT code generation

---

## Optimization Targets

### 1. Expression Compilation

**Target**: `ExprAst` (in `crates/grafial-frontend/src/ast.rs:45`)

Currently interpreted via `eval_expr()` with dynamic dispatch. Compile to native code:

```rust
pub enum ExprAst {
    Number(f64),
    Bool(bool),
    Binary { op: BinaryOp, left: Box<ExprAst>, right: Box<ExprAst> },
    Call { name: String, args: Vec<CallArg> },
    // ... more variants
}
```

**LLVM Benefits**:
- Constant folding: `2.0 + 3.0` → `5.0` at compile time
- Inlining: Small function calls eliminated
- SIMD vectorization: Batch operations on arrays
- Dead code elimination: Unreachable branches removed

**Example Use Case**: Metric computations in flows that run over thousands of nodes.

### 2. Rule Pattern Matching

**Target**: `RuleDef` patterns (in `crates/grafial-frontend/src/ast.rs:110`)

```rust
pub struct RuleDef {
    pub patterns: Vec<PatternItem>,     // (src)-[edge]->(dst)
    pub where_expr: Option<ExprAst>,    // Filter predicate
    pub actions: Vec<ActionStmt>,       // Modifications
}
```

**Current Approach**: Runtime pattern matching via query plans
**LLVM Approach**: Compile patterns to specialized matching functions

**Benefits**:
- Hardcoded adjacency checks (no vtable lookups)
- Branch prediction optimization for common patterns
- Loop unrolling for fixed-size patterns

### 3. Flow Transformations

**Target**: `FlowDef` pipelines (in `crates/grafial-frontend/src/ast.rs:155`)

```rust
pub enum Transform {
    ApplyRule { rule: String },
    ApplyRuleset { rules: Vec<String> },
    PruneEdges { edge_type: String, predicate: ExprAst },
}
```

**LLVM Approach**: Compile entire transformation pipelines

**Benefits**:
- Fuse multiple transformations (eliminate intermediate graphs)
- Optimize rule application order
- Specialize code for specific schema types

### 4. Bayesian Computations

**Target**: Posterior updates in belief graphs

Current conjugate update formulas (Gaussian, Beta, Dirichlet) can be compiled to tight loops with LLVM:

**Benefits**:
- Vectorize batch updates (SIMD)
- Fuse multiple updates
- Cache-friendly memory access patterns

---

## Implementation Approach

### Phase 1: Infrastructure Setup

**Crate**: Create `crates/grafial-llvm/`

**Dependencies**:
```toml
[dependencies]
inkwell = { version = "0.4", features = ["llvm17-0"] }
grafial-ir = { path = "../grafial-ir" }
grafial-frontend = { path = "../grafial-frontend" }
```

**Key Structs**:
```rust
pub struct LLVMCompiler<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: Option<ExecutionEngine<'ctx>>,
}
```

### Phase 2: Expression Compiler (Quick Win)

**Goal**: Compile `ExprAst` to LLVM IR

**Input**: Expression from `where` clause or metric definition
**Output**: Native function pointer

**Example**:
```rust
// Grafial: where v1.conversion_rate > 0.1
// AST: Binary { op: GreaterThan, left: Field{...}, right: Number(0.1) }
// LLVM IR:
define double @expr_gt_0(%Node* %v1) {
  %1 = getelementptr %Node, %Node* %v1, i32 0, i32 1
  %2 = load double, double* %1
  %3 = fcmp ogt double %2, 0.1
  %4 = uitofp i1 %3 to double
  ret double %4
}
```

**Implementation**:
```rust
impl<'ctx> LLVMCompiler<'ctx> {
    pub fn compile_expr(&mut self, expr: &ExprAst) -> FunctionValue<'ctx> {
        match expr {
            ExprAst::Number(n) => self.builder.build_float_constant(*n),
            ExprAst::Binary { op, left, right } => {
                let lhs = self.compile_expr(left);
                let rhs = self.compile_expr(right);
                match op {
                    BinaryOp::Add => self.builder.build_float_add(lhs, rhs, "add"),
                    BinaryOp::Mul => self.builder.build_float_mul(lhs, rhs, "mul"),
                    // ... more ops
                }
            }
            // ... more variants
        }
    }
}
```

### Phase 3: Rule Compilation

**Goal**: Compile pattern matching + actions to native code

**Challenges**:
- Pattern matching is inherently dynamic (graph structure varies)
- Need efficient encoding of graph data for LLVM

**Strategy**: Hybrid Approach
- **Interpreted**: Graph traversal (too dynamic for LLVM)
- **Compiled**: Predicate evaluation + actions (hot path)

**Example**:
```rust
// Grafial rule:
// rule PropagateBeliefs on TestBeliefs {
//   pattern (v1:Variant)-[e:OUTPERFORMS]->(v2:Variant)
//   where v1.conversion_rate > v2.conversion_rate
//   action { delete edge e }
// }

// Compiled predicate function:
define i1 @rule_predicate_0(
    %Node* %v1,
    %Edge* %e,
    %Node* %v2
) {
entry:
  %v1_cr = getelementptr %Node, %Node* %v1, i32 0, i32 1
  %v1_val = load double, double* %v1_cr
  %v2_cr = getelementptr %Node, %Node* %v2, i32 0, i32 1
  %v2_val = load double, double* %v2_cr
  %cmp = fcmp ogt double %v1_val, %v2_val
  ret i1 %cmp
}
```

### Phase 4: Flow AOT Compilation

**Goal**: Compile entire flows to standalone functions

**Benefits**:
- Maximum optimization (whole-program analysis)
- No interpreter overhead
- Deployment as shared libraries

**Example**:
```rust
// Grafial flow:
// flow Demo on TestBeliefs {
//   graph g = from_evidence VariantBData
//           | apply_rule PropagateBeliefs
//   metric accuracy = count_edges(g, "OUTPERFORMS")
// }

// Compiled flow:
define double @flow_demo(%Evidence* %evidence) {
entry:
  %g0 = call %Graph* @load_evidence(%Evidence* %evidence)
  %g1 = call %Graph* @apply_rule_0(%Graph* %g0)
  %count = call i64 @count_edges(%Graph* %g1, i8* getelementptr([11 x i8], [11 x i8]* @str_outperforms, i32 0, i32 0))
  %result = uitofp i64 %count to double
  ret double %result
}
```

---

## Code Examples

### Example 1: Setting Up LLVM Context

```rust
use inkwell::context::Context;
use inkwell::OptimizationLevel;

pub fn create_llvm_compiler() -> LLVMCompiler<'static> {
    let context = Context::create();
    let module = context.create_module("grafial");
    let builder = context.create_builder();

    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .expect("Failed to create execution engine");

    LLVMCompiler {
        context: &context,
        module,
        builder,
        execution_engine: Some(execution_engine),
    }
}
```

### Example 2: Compiling Simple Arithmetic Expression

```rust
impl<'ctx> LLVMCompiler<'ctx> {
    pub fn compile_add_expr(&mut self) -> JitFunction<AddFunc> {
        // Create function signature: double add(double, double)
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
        let function = self.module.add_function("add", fn_type, None);

        // Create entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Get parameters
        let lhs = function.get_nth_param(0).unwrap().into_float_value();
        let rhs = function.get_nth_param(1).unwrap().into_float_value();

        // Build add instruction
        let result = self.builder.build_float_add(lhs, rhs, "add").unwrap();

        // Return result
        self.builder.build_return(Some(&result)).unwrap();

        // JIT compile and get function pointer
        unsafe {
            self.execution_engine
                .as_ref()
                .unwrap()
                .get_function("add")
                .unwrap()
        }
    }
}

type AddFunc = unsafe extern "C" fn(f64, f64) -> f64;
```

### Example 3: Optimizing with LLVM Passes

```rust
use inkwell::passes::{PassManager, PassManagerBuilder};

pub fn optimize_module(module: &Module) {
    let pass_manager_builder = PassManagerBuilder::create();
    pass_manager_builder.set_optimization_level(OptimizationLevel::Aggressive);

    let pass_manager = PassManager::create(());
    pass_manager_builder.populate_module_pass_manager(&pass_manager);

    // Add specific passes
    pass_manager.add_instruction_combining_pass();
    pass_manager.add_reassociate_pass();
    pass_manager.add_gvn_pass();
    pass_manager.add_cfg_simplification_pass();
    pass_manager.add_basic_alias_analysis_pass();
    pass_manager.add_promote_memory_to_register_pass();
    pass_manager.add_instruction_simplify_pass();

    // Run optimization passes
    pass_manager.run_on(module);
}
```

### Example 4: Compiling Grafial Expression

```rust
impl<'ctx> LLVMCompiler<'ctx> {
    pub fn compile_grafial_expr(
        &mut self,
        expr: &ExprAst,
        context_type: StructType<'ctx>,
    ) -> FunctionValue<'ctx> {
        let f64_type = self.context.f64_type();
        let ctx_ptr = context_type.ptr_type(AddressSpace::default());

        // Function signature: double eval(Context* ctx)
        let fn_type = f64_type.fn_type(&[ctx_ptr.into()], false);
        let function = self.module.add_function("eval_expr", fn_type, None);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let ctx_param = function.get_nth_param(0).unwrap().into_pointer_value();
        let result = self.compile_expr_recursive(expr, ctx_param, context_type);

        self.builder.build_return(Some(&result)).unwrap();

        function
    }

    fn compile_expr_recursive(
        &mut self,
        expr: &ExprAst,
        ctx: PointerValue<'ctx>,
        ctx_type: StructType<'ctx>,
    ) -> FloatValue<'ctx> {
        match expr {
            ExprAst::Number(n) => {
                self.context.f64_type().const_float(*n)
            }

            ExprAst::Var(name) => {
                // Look up variable in context
                let field_idx = self.get_field_index(name);
                let field_ptr = self.builder.build_struct_gep(
                    ctx_type,
                    ctx,
                    field_idx,
                    &format!("{}_ptr", name)
                ).unwrap();
                self.builder.build_load(
                    self.context.f64_type(),
                    field_ptr,
                    name
                ).unwrap().into_float_value()
            }

            ExprAst::Binary { op, left, right } => {
                let lhs = self.compile_expr_recursive(left, ctx, ctx_type);
                let rhs = self.compile_expr_recursive(right, ctx, ctx_type);

                match op {
                    BinaryOp::Add => self.builder.build_float_add(lhs, rhs, "add").unwrap(),
                    BinaryOp::Sub => self.builder.build_float_sub(lhs, rhs, "sub").unwrap(),
                    BinaryOp::Mul => self.builder.build_float_mul(lhs, rhs, "mul").unwrap(),
                    BinaryOp::Div => self.builder.build_float_div(lhs, rhs, "div").unwrap(),
                    BinaryOp::GreaterThan => {
                        let cmp = self.builder.build_float_compare(
                            inkwell::FloatPredicate::OGT,
                            lhs,
                            rhs,
                            "gt"
                        ).unwrap();
                        self.builder.build_unsigned_int_to_float(
                            cmp,
                            self.context.f64_type(),
                            "gt_float"
                        ).unwrap()
                    }
                    // ... more operators
                    _ => panic!("Unsupported operator: {:?}", op),
                }
            }

            ExprAst::Call { name, args } => {
                // Compile function call (simplified)
                match name.as_str() {
                    "sqrt" => {
                        let arg = self.compile_expr_recursive(&args[0].expr, ctx, ctx_type);
                        let sqrt_fn = self.module.get_function("llvm.sqrt.f64")
                            .unwrap_or_else(|| {
                                let fn_type = self.context.f64_type().fn_type(
                                    &[self.context.f64_type().into()],
                                    false
                                );
                                self.module.add_function("llvm.sqrt.f64", fn_type, None)
                            });
                        self.builder.build_call(sqrt_fn, &[arg.into()], "sqrt")
                            .unwrap()
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_float_value()
                    }
                    _ => panic!("Unknown function: {}", name),
                }
            }

            _ => panic!("Unsupported expression: {:?}", expr),
        }
    }
}
```

### Example 5: Integration with Grafial Core

```rust
// In crates/grafial-core/src/engine/expr_eval.rs

use grafial_llvm::LLVMCompiler;

pub struct CompiledExprCache<'ctx> {
    compiler: LLVMCompiler<'ctx>,
    cache: HashMap<ExprAst, JitFunction<'ctx, ExprFunc>>,
}

type ExprFunc = unsafe extern "C" fn(*const ExprContext) -> f64;

impl<'ctx> CompiledExprCache<'ctx> {
    pub fn eval_expr(&mut self, expr: &ExprAst, ctx: &dyn ExprContext) -> f64 {
        // Check cache
        if let Some(jit_fn) = self.cache.get(expr) {
            unsafe { jit_fn.call(ctx as *const _ as *const ExprContext) }
        } else {
            // Compile and cache
            let function = self.compiler.compile_grafial_expr(expr, /* context_type */);
            let jit_fn = unsafe {
                self.compiler.execution_engine
                    .as_ref()
                    .unwrap()
                    .get_function("eval_expr")
                    .unwrap()
            };
            self.cache.insert(expr.clone(), jit_fn);
            unsafe { jit_fn.call(ctx as *const _ as *const ExprContext) }
        }
    }
}
```

---

## Performance Considerations

### When to Use LLVM

**Good Use Cases**:
- ✅ Hot loops (metric computations over 10k+ nodes)
- ✅ Complex mathematical expressions (Bayesian updates)
- ✅ Repeated rule applications (fixpoint iterations)
- ✅ Production deployments (AOT compilation)

**Poor Use Cases**:
- ❌ One-off expressions (compilation overhead > runtime savings)
- ❌ Highly dynamic patterns (graph structure varies too much)
- ❌ Development/debugging (slower iteration cycles)

### Compilation Overhead

**JIT Compilation Time**:
- Simple expression: ~1-5ms
- Complex rule: ~10-50ms
- Full flow: ~100-500ms

**Breakeven Point**: Expression must execute >1000 times to amortize compilation cost

**Mitigation**:
- Cache compiled functions
- Use tiered compilation (interpret first, JIT hot paths)
- Persist compiled code to disk

### Memory Overhead

**LLVM Infrastructure**:
- Context: ~10MB base overhead
- Per-function: ~1-10KB LLVM IR + native code

**Mitigation**:
- Share Context across compilations
- Lazy compilation (compile on first use)
- Unload cold functions

### Optimization Trade-offs

**Aggressive Optimization** (`-O3`):
- **Pros**: 2-10x speedup, vectorization, inlining
- **Cons**: Slower compilation (2-5x), larger code size

**Recommendation**:
- Development: `-O1` or interpret
- Production: `-O3` with AOT compilation

---

## Integration Roadmap

### Milestone 1: Expression JIT (2-3 weeks)
- [ ] Create `crates/grafial-llvm/`
- [ ] Implement basic expression compiler
- [ ] Add JIT cache to `expr_eval.rs`
- [ ] Benchmark on metric computations
- [ ] **Goal**: 2-5x speedup on metric-heavy flows

### Milestone 2: Rule Predicate Compilation (3-4 weeks)
- [ ] Compile `where` clauses to LLVM
- [ ] Integrate with `rule_exec.rs`
- [ ] Support field access and function calls
- [ ] **Goal**: 3-10x speedup on filter-heavy rules

### Milestone 3: Bayesian Computation Vectorization (2-3 weeks)
- [ ] Compile posterior update formulas
- [ ] Add SIMD intrinsics for batch updates
- [ ] Integrate with `BeliefGraph`
- [ ] **Goal**: 5-20x speedup on evidence ingestion

### Milestone 4: Flow AOT Compilation (4-6 weeks)
- [ ] Whole-flow IR lowering
- [ ] Static compilation to shared libraries
- [ ] CLI integration (`grafial compile flow.gf -o flow.so`)
- [ ] **Goal**: Deploy optimized flows without interpreter

---

## References

### LLVM Resources
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- [inkwell Documentation](https://thedan64.github.io/inkwell/)
- [Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/) - Implementing a language with LLVM

### Rust LLVM Examples
- [rustc LLVM Codegen](https://github.com/rust-lang/rust/tree/master/compiler/rustc_codegen_llvm)
- [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) - Alternative fast compiler
- [Inkwell Examples](https://github.com/TheDan64/inkwell/tree/master/examples)

### Related Projects
- [GraalVM Truffle](https://www.graalvm.org/latest/graalvm-as-a-platform/language-implementation-framework/) - AST interpreter framework with JIT
- [PyPy](https://www.pypy.org/) - Python JIT compiler
- [LuaJIT](https://luajit.org/) - Lua JIT compiler with excellent performance

### Bayesian Computation Optimization
- [Stan Math Library](https://github.com/stan-dev/math) - Optimized Bayesian computations
- [SIMD for Statistical Computing](https://arxiv.org/abs/1809.02982)

---

## Appendix: Alternative Approaches

### Cranelift (Simpler Alternative)

If LLVM proves too complex, consider **Cranelift**:
- Faster compilation (10-100x vs LLVM)
- Simpler API
- Good enough optimization for most cases
- Used by Wasmtime/SpiderMonkey

**Trade-off**: ~30-50% slower runtime performance vs LLVM

### Tiered Compilation

Hybrid approach:
1. **Tier 0**: Interpret AST (immediate execution)
2. **Tier 1**: Cranelift JIT (fast compilation, good perf)
3. **Tier 2**: LLVM JIT (slow compilation, best perf)

Automatically promote hot code paths to higher tiers.

### Partial Evaluation

Instead of full compilation, specialize interpreter:
- Constant fold known values
- Inline small expressions
- Cache predicate results

**Benefits**: Simpler implementation, still 2-3x speedup

---

## Conclusion

LLVM offers significant optimization opportunities for Grafial:

1. **Quick Win**: Expression JIT for metrics (2-5x speedup)
2. **Medium Win**: Rule predicate compilation (3-10x speedup)
3. **Big Win**: Full flow AOT compilation (10-100x speedup)

**Recommended Strategy**:
- Start with expression JIT (high ROI, low complexity)
- Measure impact on real-world workloads
- Incrementally add rule and flow compilation
- Consider Cranelift if LLVM complexity becomes a blocker

The key is leveraging Grafial's existing IR infrastructure (`grafial-ir`) as the lowering target for LLVM codegen, ensuring clean separation between frontend, optimization, and execution layers.
