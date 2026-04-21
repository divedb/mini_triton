# Triton-Like Development Plan

## Goal

Evolve `mini_triton` from the current vector-add vertical slice into a Triton-like GPU kernel compiler/runtime project while keeping the codebase understandable and incrementally testable.

## Scope and Principles

1. Keep one end-to-end path green at all times.
2. Add capabilities in thin vertical increments, not broad rewrites.
3. Preserve explicit compiler stages and inspectable artifacts.
4. Optimize only after correctness and observability are in place.

## Current Baseline

The project currently has:

1. Python kernel capture for a restricted model.
2. Internal IR and MLIR emission.
3. C++ lowering driver integration.
4. PTX generation and CUDA launch path.
5. Working vector-add execution examples and smoke tests.

This baseline is the control path for all future changes.

## Phased Roadmap

## Phase 1: Baseline Hardening

Objective:
Stabilize the current vertical slice to reduce churn while adding new features.

Deliverables:

1. Reliable compile+run scripts for Windows setup.
2. Golden artifact snapshots for one kernel.
3. Test coverage for compile, cache, and runtime launch.

Acceptance Criteria:

1. `examples/vector_add.py` is reproducibly passing.
2. `python -m pytest -q` has no regressions.

## Phase 2: Triton-Like Python Surface

Objective:
Introduce a kernel authoring API closer to Triton idioms.

Deliverables:

1. Program index primitives, masked loads/stores, and vector/block ops.
2. Compile-time constants support for launch parameters.
3. Clear error messages for unsupported constructs.

Acceptance Criteria:

1. At least 3 kernel examples use the new API shape.
2. Existing vector-add remains supported.

## Phase 3: IR Upgrade for Tensor/Block Semantics

Objective:
Represent shaped values and vectorized operations in internal IR.

Deliverables:

1. IR node support for block pointers/arange-like indexing.
2. Type promotion and broadcasting rules.
3. Predication/masking semantics captured in IR directly.

Acceptance Criteria:

1. IR can express elementwise and reduction-ready patterns.
2. IR serialization remains deterministic.

## Phase 4: Compiler Pass Framework

Objective:
Move from single-step lowering to structured optimization passes.

Deliverables:

1. Pass manager abstraction.
2. Canonicalization, DCE, CSE, and shape simplification passes.
3. Optional dump points for pass-by-pass debugging.

Acceptance Criteria:

1. Same input kernel generates stable output across runs.
2. Passes have unit tests and no behavior regressions.

## Phase 5: Dialect and MLIR Lowering Consolidation

Objective:
Strengthen the C++ pipeline from custom dialect to LLVM/NVVM-ready MLIR.

Deliverables:

1. Expanded custom dialect parsing/validation.
2. Lowering coverage for additional arithmetic and control patterns.
3. Better diagnostics on dialect parse/lower failures.

Acceptance Criteria:

1. New kernel families lower without ad hoc branching.
2. Failure messages identify exact IR/dialect cause.

## Phase 6: Runtime Capability Expansion

Objective:
Approach practical Triton-like runtime behavior.

Deliverables:

1. Stream support and launch options.
2. Module/function reuse cache in runtime.
3. Optional interoperability hooks for external tensor buffers.

Acceptance Criteria:

1. Repeated launches avoid repeated module loads.
2. Runtime API supports asynchronous usage patterns.

## Phase 7: Kernel Coverage Expansion

Objective:
Support more than elementwise add.

Deliverables:

1. Unary map kernels.
2. Simple reductions.
3. Initial block GEMM prototype.

Acceptance Criteria:

1. Each kernel family has reference-checked tests.
2. Basic perf sanity checks exist for each family.

## Phase 8: Performance and Autotuning

Objective:
Improve throughput and launch configuration quality.

Deliverables:

1. Autotuning loop for block size / launch params.
2. Per-architecture cache keys.
3. Benchmark harness for regression tracking.

Acceptance Criteria:

1. Tuned mode outperforms fixed default on representative workloads.
2. Performance regressions are detectable in CI.

## Near-Term Execution Plan (Next 2 Weeks)

1. Stabilize tests and baseline scripts from Phase 1.
2. Add Triton-like API primitives (Phase 2) for masked block elementwise.
3. Extend IR to carry vector/block operations (Phase 3).
4. Introduce initial pass manager with canonicalization + DCE (Phase 4).
5. Add one additional kernel example beyond vector add.

## Risks and Mitigations

1. Risk: Feature growth outpaces test reliability.
Mitigation: Require tests and artifact checks for each feature increment.

2. Risk: Dialect/MLIR divergence makes debugging hard.
Mitigation: Keep deterministic dumps and staged lowering boundaries.

3. Risk: Runtime complexity blocks compiler progress.
Mitigation: Isolate runtime enhancements behind narrow interfaces.

## Definition of Success

The plan is successful when `mini_triton` can author, compile, and run multiple Triton-like kernels with stable compiler behavior, clear diagnostics, reproducible artifacts, and measurable performance progress.
