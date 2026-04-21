# mini_triton

## Project Framing

`mini_triton` is a clean-room, inspiration-based CUDA kernel compiler stack that borrows high-level goals from Triton-style systems without cloning Triton source code, file layout, symbols, or implementation details. The purpose of this repository is to build an original, minimal, understandable system that offers a Python-facing kernel programming model and uses MLIR plus the local LLVM toolchain to lower work to CUDA-compatible GPU code.

In this project, "Triton-like" means the following and nothing more for the initial scope:

- Python is the public authoring and invocation layer.
- Kernel capture and compilation flow are explicit and programmable from Python.
- MLIR is the central compiler substrate for IR representation, validation, lowering, and code generation staging.
- The backend path targets CUDA through LLVM's NVPTX path and the CUDA driver/runtime stack.
- The first usable version behaves like a simple on-demand compiler with cacheable outputs; it does not need a sophisticated production JIT.

The MVP is intentionally narrow. It does not attempt to reproduce a full GPU DSL, autotuning system, advanced schedule search, layout algebra, fusion engine, block-pointer abstractions, tensor core support, multi-axis launch sophistication, or a large standard library. The first goal is a vertical slice that proves the architecture end to end.

## Project Goal

Build a minimal but real CUDA kernel compiler stack with these properties:

- A small Python kernel definition API.
- A restricted symbolic kernel capture layer.
- A compiler path that produces MLIR and lowers it toward LLVM IR / PTX.
- A Python runtime path that loads generated GPU code and launches kernels.
- Enough functionality to execute a small elementwise kernel on CUDA tensors or device buffers.

## Non-Goals

The following are explicitly out of scope for the MVP:

- Full parity with Triton semantics or API design.
- Arbitrary Python control-flow capture.
- Autotuning and schedule search.
- Fusion across kernels.
- Complex layout transformations.
- Shared-memory tiling beyond what is strictly required for a tiny example.
- Tensor core or WMMA support.
- Multi-GPU execution.
- Production-grade error diagnostics and profiling.
- Broad dtype and shape coverage.

## Architecture Overview

The system is organized as a narrow vertical pipeline:

1. Python kernel authoring
   - Users write a restricted Python function against a tiny DSL.
2. Frontend capture
   - The function is executed against symbolic proxy objects to build a small internal kernel graph.
3. MLIR emission
   - The internal graph is lowered into a narrow MLIR form, initially using the smallest viable combination of standard MLIR dialects and, only if needed later, a custom dialect for missing semantics.
4. Lowering and code generation
   - MLIR is lowered through a controlled pass pipeline toward LLVM-compatible GPU code, then translated to LLVM IR and PTX using the local LLVM/MLIR build.
5. Runtime and launch
   - Python loads the generated PTX, prepares launch arguments, and dispatches the CUDA kernel.
6. Caching and validation
   - Compiled artifacts are cached by signature so repeated runs do not rebuild unnecessarily.

The key design choice is to keep the frontend capture language smaller than general Python and keep the compiler pipeline explicit. The MVP will favor a direct, inspectable implementation over magic.

## What The MVP Supports

The first working version should support only a very small kernel family:

- One-dimensional launch geometry.
- Contiguous 1D buffers.
- `float32` inputs and outputs only.
- A single elementwise binary kernel: add.
- A single elementwise unary map shape as the next extension point.
- A fixed or lightly parameterized block size.
- Predicated out-of-bounds handling for tail elements.
- On-demand compile plus cache on first invocation.

The intended first user-visible demo is conceptually:

- Allocate CUDA buffers.
- Define a Python kernel that reads `x[i]` and `y[i]`, computes `x[i] + y[i]`, and writes `out[i]`.
- Compile on first call.
- Launch on CUDA.
- Compare against a host-side reference.

## Major Components

### 1. Python API Layer

Responsibilities:

- Expose the kernel decorator or builder entry point.
- Define the restricted kernel authoring model.
- Accept launch parameters and argument metadata.
- Trigger compilation and artifact caching.

Initial design direction:

- Prefer a tiny API based on explicit context objects rather than copying Triton's exact surface syntax.
- Kernel functions should receive a context plus typed buffer arguments.
- The kernel body should be evaluated symbolically, not compiled from arbitrary Python bytecode.

Example shape for discussion only:

```python
@mt.kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_indices()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)
```

This example is illustrative. It defines the style of API the project should aim for, not a locked public contract.

### 2. Frontend Capture Layer

Responsibilities:

- Provide symbolic values for indices, masks, loads, stores, and arithmetic.
- Record a small SSA-like internal graph from Python evaluation.
- Reject unsupported Python constructs early.

Initial constraint:

- MVP capture is API-driven symbolic evaluation, not a general-purpose AST compiler.
- AST inspection may still be used for validation and better diagnostics, but it is not required for the first vertical slice.

### 3. Internal Kernel IR

Responsibilities:

- Represent the small set of kernel operations independent of Python syntax.
- Normalize the captured graph into a deterministic compiler input.
- Carry type, shape, pointer-space, and launch metadata.

Initial node categories:

- Constants and scalar parameters.
- Program index computation.
- Predicates and comparisons.
- Elementwise arithmetic.
- Buffer load/store with mask.

This IR is intentionally tiny and exists only to simplify the boundary between Python capture and MLIR generation.

### 4. MLIR Compiler Layer

Responsibilities:

- Materialize the kernel IR in MLIR.
- Run structural verification and canonicalization.
- Lower toward LLVM-compatible GPU code.

MVP strategy:

- Start with the smallest viable MLIR representation.
- Prefer standard dialects where practical.
- Introduce a custom dialect only if the frontend semantics become awkward or lossy when expressed directly in existing MLIR dialects.

Probable early MLIR stack:

- `builtin`
- `func`
- `arith`
- `cf` or `scf`
- `memref` or LLVM-pointer-oriented representation, depending on runtime ABI choices
- `llvm`
- `nvvm` where needed for CUDA-targeted code generation

The MVP does not need a rich custom dialect if direct lowering from a small internal IR into standard MLIR is manageable.

### 5. CUDA Codegen Path

Responsibilities:

- Turn MLIR into GPU-targetable LLVM IR.
- Use the local LLVM backend to emit PTX.
- Produce an artifact that the runtime can load.

Practical MVP route:

1. Emit MLIR in a form translatable to LLVM IR.
2. Use the local `mlir-translate` to obtain LLVM IR.
3. Use the local `llc` with NVPTX target settings to emit PTX.
4. Load PTX through a Python CUDA runtime binding.

This route is intentionally conservative. It avoids over-committing to more advanced MLIR GPU binary serialization paths before the base pipeline works.

### 6. Runtime Layer

Responsibilities:

- Manage CUDA module loading.
- Marshal kernel arguments.
- Compute launch dimensions.
- Launch kernels and surface runtime failures.

Initial design direction:

- Use a Python-level CUDA binding with a low enough abstraction level to load PTX and launch kernels.
- Keep the runtime ABI narrow: raw device pointers, scalar sizes, and launch configuration.
- Defer richer tensor objects until the compiler and launch path are stable.

### 7. Test and Example Layer

Responsibilities:

- Verify end-to-end compilation.
- Verify runtime correctness on one or two kernels.
- Keep regression tests aligned with the small MVP scope.

## IR Levels And Lowering Stages

The compiler pipeline should be explicit about IR levels:

1. Python authoring IR
   - Restricted Python DSL invocation.
2. Captured kernel graph
   - Small internal SSA-like graph.
3. Kernel MLIR
   - MLIR module representing kernel semantics and ABI.
4. LLVM-oriented MLIR
   - After canonicalization and lowering into LLVM/NVVM-friendly form.
5. LLVM IR
   - Produced by `mlir-translate`.
6. PTX
   - Produced by `llc` targeting NVPTX.
7. Loaded CUDA module
   - Runtime artifact ready for execution.

Early passes should stay simple:

- Type checking.
- Canonicalization.
- Dead code cleanup where trivial.
- Lowering of masked elementwise operations into explicit control/data flow.
- ABI materialization for kernel entry points.

## Directory Layout Proposal

This layout is a proposal for the first phases, not a commitment to fill every directory immediately:

```text
mini_triton/
  README.md
  CMakeLists.txt
  pyproject.toml
  cmake/
    LLVMConfigHelpers.cmake
  python/
    mini_triton/
      __init__.py
      api.py
      capture.py
      ir.py
      compiler.py
      runtime.py
      cache.py
  tools/
    mtc-translate/
      CMakeLists.txt
      main.cpp
  include/
    mini_triton/
      Compiler/
  lib/
    Compiler/
  test/
    python/
    mlir/
    end_to_end/
  examples/
    vector_add.py
  third_party/
    llvm-project/
```

Notes:

- `tools/mtc-translate` is a placeholder name for a tiny compiler-side executable if Python bindings alone are not enough. It should not be created until Phase 3 or Phase 4 proves it is necessary.
- `include/` and `lib/` may remain sparse early on if the first compiler driver is small.
- The Python package should remain the primary control plane.

## Build And Integration Assumptions

The repository assumes a local `llvm-project` checkout already exists and that LLVM/MLIR has already been built somewhere on disk. The project should reuse that existing build instead of telling the user to fetch or rebuild LLVM.

The build design should support configuration through explicit paths:

- `MINITRITON_LLVM_PROJECT_DIR`
  - Path to the local `llvm-project` source tree.
- `MINITRITON_LLVM_BUILD_DIR`
  - Path to the existing LLVM/MLIR build tree.
- `MLIR_DIR`
  - Usually `${MINITRITON_LLVM_BUILD_DIR}/lib/cmake/mlir`.
- `LLVM_DIR`
  - Usually `${MINITRITON_LLVM_BUILD_DIR}/lib/cmake/llvm`.

The implementation should detect or validate these paths rather than assuming a fixed machine-specific layout.

Expected reusable local tools:

- `mlir-opt`
- `mlir-translate`
- `llc`
- optionally `clang` for host-side experiments

The project should prefer linking against the local MLIR/LLVM libraries only where a small in-repo compiler tool is actually justified. For the MVP, command-line tool orchestration is acceptable if it reduces moving parts.

## Quick Start (Current MVP)

This project currently supports a compile-first MVP flow in Python:

1. Capture a restricted Python kernel.
2. Emit MLIR.
3. Compile to PTX with local `mlir-opt`, `mlir-translate`, and `llc`.
4. Reuse cached PTX on repeated compile requests with the same kernel signature and CUDA arch.

Assuming `.venv` already exists and LLVM/MLIR is already built locally:

```powershell
cd e:\mini_triton
.\.venv\Scripts\python.exe -m pip install -e .
set MINITRITON_LLVM_BUILD_DIR=e:\mini_triton\third_party\llvm-project\build
.\.venv\Scripts\Activate.ps1
python examples\compile_vector_add.py
```

Run compile + launch directly and print vector-add result:

```powershell
python examples\vector_add.py
```

Run Python tests:

```powershell
python -m pytest -q
```

CLI compile path:

```powershell
python -m mini_triton.cli compile-add --llvm-build-dir e:\mini_triton\third_party\llvm-project\build
```

Optional C++ lowering driver build (keeps Python as control plane, moves lowering orchestration into a C++ executable):

```powershell
cmake -S . -B build\host -DMINITRITON_LLVM_BUILD_DIR=e:\mini_triton\third_party\llvm-project\build
cmake --build build\host --config Release --target mini_triton_lower
```

Use the C++ lowering driver for compile-add (auto-detected when built under `build/host`):

```powershell
python -m mini_triton.cli compile-add `
   --llvm-build-dir e:\mini_triton\third_party\llvm-project\build
```

If the executable is in a non-default location, set `MINITRITON_CPP_LOWERING_DRIVER` or pass `--cpp-lowering-driver`.

CLI cache management:

```powershell
python -m mini_triton.cli cache-list --output-dir build\mvp_artifacts
python -m mini_triton.cli cache-prune --output-dir build\mvp_artifacts --keep 5
```

CLI toolchain inspection:

```powershell
python -m mini_triton.cli toolchain-check --llvm-build-dir e:\mini_triton\third_party\llvm-project\build\Release
```

CLI runtime launch smoke test:

```powershell
python -m mini_triton.cli launch-smoke --ptx-path build\gpu_smoke\noop.ptx --kernel-name noop
```

CLI full vector-add GPU smoke test:

```powershell
python -m mini_triton.cli vector-add-smoke --llvm-build-dir e:\mini_triton\third_party\llvm-project\build\Release --num-elements 4096
```

Vector-add smoke test via C++ lowering driver:

```powershell
python -m mini_triton.cli vector-add-smoke `
   --llvm-build-dir e:\mini_triton\third_party\llvm-project\build `
   --num-elements 4096
```

Notes:

- On first run, `compile_cached` compiles and writes artifacts under `build/mvp_artifacts/cache/<key>/...`.
- On subsequent runs with identical kernel IR, architecture, and toolchain fingerprint, the same PTX path is reused without recompiling.

## CUDA Backend Assumptions

The CUDA path for the MVP should be explicit and practical:

- Input buffers are already CUDA device buffers or are wrapped by a minimal runtime object that can expose raw device pointers.
- The compiler emits PTX, not necessarily cubin.
- Kernel loading uses the CUDA driver path from Python.
- The launch model is one-dimensional grid and one-dimensional block.

Real in MVP:

- PTX generation from the MLIR/LLVM pipeline.
- PTX loading and kernel launch from Python.
- Correct execution for one or two very small kernels.

Deferred or stubbed after MVP:

- Fatbin generation.
- Architecture-specific tuning.
- Shared memory management APIs.
- Stream management beyond a basic default path.
- Occupancy heuristics.

Runtime note:

- The current Python runtime default attempts to load the CUDA driver library directly (`nvcuda.dll` on Windows, `libcuda.so` on Linux) via `ctypes`.
- If the driver library is unavailable, callers can still inject a custom runtime driver implementation for testing.

## MVP First

The first working version must support exactly one practical vertical slice:

- Kernel family: elementwise vector add on contiguous 1D `float32` buffers.
- Launch model: fixed block size, computed grid size.
- Frontend: restricted Python kernel declaration using symbolic buffer operations.
- Capture: API-driven symbolic execution into a tiny internal IR.
- Compiler: internal IR to MLIR to LLVM IR to PTX using the local LLVM/MLIR build.
- Runtime: Python loads PTX, passes raw device pointers and scalar length, launches kernel.
- Validation: compare device result against a CPU reference on a small test case.

The MVP should not attempt to support general tensor semantics, broadcasting, reductions, autotuning, or advanced kernel scheduling. If a feature does not directly help the vector-add path work end to end, it should be deferred.

## Development Roadmap

## Phase 1: Minimal Architecture And Scaffolding

Objective:

- Establish repository structure, configuration conventions, and the compiler/runtime boundary without building unnecessary subsystems.

Deliverables:

- This README as the design anchor.
- Initial repository layout skeleton.
- A path configuration strategy for local LLVM/MLIR integration.
- A short decision record on whether MVP codegen is driven by Python plus external MLIR tools, a small in-repo compiler executable, or both.

Dependencies:

- Existing local LLVM/MLIR source tree and build tree.
- CUDA-capable machine and runtime libraries.

Risks:

- Over-designing the architecture before the first kernel exists.
- Tying the build to a machine-specific LLVM layout.

Validation criteria:

- The repository has a documented plan that maps directly to a minimal end-to-end implementation.
- The local LLVM/MLIR reuse assumptions are explicit and testable.

## Phase 2: Basic Python Frontend And API Capture

Objective:

- Create the smallest useful Python kernel authoring interface and symbolic capture path.

Deliverables:

- A kernel decorator or builder API.
- Symbolic value classes for indices, masks, arithmetic, loads, and stores.
- A restricted execution model that records a tiny internal graph.
- Early validation errors for unsupported constructs.

Dependencies:

- Phase 1 design decisions.
- A minimal runtime ABI definition for buffer arguments and scalar parameters.

Risks:

- Making the Python API too magical or too close to Triton's exact idioms.
- Letting unsupported Python semantics leak into the capture path.

Validation criteria:

- A simple vector-add kernel can be captured deterministically into an internal representation.
- Unsupported control flow or unsupported operations fail early with clear errors.

## Phase 3: MLIR Representation And Verification

Objective:

- Convert the captured kernel graph into a small, valid MLIR module and verify the representation strategy.

Deliverables:

- MLIR emission for the vector-add kernel.
- A documented choice between direct use of standard dialects and introducing a tiny custom dialect.
- Basic verification and canonicalization support.
- Golden MLIR tests for one or two kernels.

Dependencies:

- Captured internal IR from Phase 2.
- Access to local MLIR tooling or a small parser/driver linked against MLIR.

Risks:

- Premature custom dialect design.
- Picking an MLIR encoding that makes later lowering awkward.

Validation criteria:

- The compiler can generate MLIR for vector add.
- The generated MLIR passes validation and can be inspected as a stable intermediate artifact.

## Phase 4: Lowering Pipeline Toward LLVM And CUDA

Objective:

- Establish the real compile path from MLIR to PTX using the local LLVM/MLIR build.

Deliverables:

- A pass pipeline that lowers kernel MLIR into LLVM/NVVM-compatible form.
- Translation to LLVM IR.
- PTX generation via NVPTX backend.
- Artifact capture for debugging, including optional saved MLIR, LLVM IR, and PTX files.

Dependencies:

- Phase 3 MLIR output.
- Working `mlir-translate` and `llc` from the local LLVM build.
- CUDA target details such as compute capability selection.

Risks:

- ABI mismatches between generated code and runtime launch.
- Choosing a lowering path that depends on unproven MLIR GPU features for the MVP.

Validation criteria:

- The vector-add kernel compiles into PTX reproducibly.
- Generated PTX contains a callable kernel entry point with the expected parameters.

## Phase 5: Runtime Integration And Kernel Launch

Objective:

- Load compiled GPU code from Python and execute the first real kernel.

Deliverables:

- A minimal runtime wrapper for CUDA module loading and launch.
- Argument marshaling for device pointers and scalar sizes.
- Launch configuration helper for one-dimensional kernels.
- A compilation cache keyed by kernel identity and launch-relevant metadata.

Dependencies:

- PTX generation from Phase 4.
- Available Python CUDA binding or a minimal driver bridge.

Risks:

- Runtime pointer ownership confusion.
- Cache invalidation bugs.
- Silent launch failures if diagnostics are too weak.

Validation criteria:

- A Python script can compile, launch, and verify vector add on CUDA.
- Re-running the same kernel reuses cached artifacts when appropriate.

## Phase 6: Testing, Examples, And Incremental Improvements

Objective:

- Stabilize the vertical slice and add only the next smallest features that preserve clarity.

Deliverables:

- Unit tests for capture and IR generation.
- End-to-end tests for vector add and one unary map kernel.
- Example scripts that demonstrate compilation artifacts and runtime behavior.
- Limited extension of dtype coverage or API ergonomics only after the MVP path is solid.

Dependencies:

- Working end-to-end pipeline from Phase 5.

Risks:

- Expanding scope before the first path is stable.
- Adding too many features without strengthening tests.

Validation criteria:

- End-to-end tests pass consistently on the intended machine.
- The repository contains one clear example and one clear test path for the first supported kernels.

## Sequential Implementation Plan Derived From The Roadmap

After the README is accepted, implementation should proceed in small, ordered tasks:

1. Add minimal project metadata and configuration files.
2. Add the Python package skeleton with only the kernel declaration and capture primitives needed for vector add.
3. Add a tiny internal IR with deterministic printing for tests.
4. Add MLIR emission for the internal IR.
5. Add a simple compile driver that invokes the local MLIR/LLVM tools.
6. Add the smallest runtime path that can load PTX and launch one kernel.
7. Add one vector-add example and one correctness test.
8. Only then add one more kernel shape, such as unary pointwise map.

This order matters. The repository should always prefer one complete end-to-end path over multiple half-built abstractions.

## Design Principles For Implementation

- Keep each module small and explainable.
- Avoid hidden compiler magic.
- Keep artifacts inspectable at every stage.
- Prefer explicit configuration over auto-detection that is hard to debug.
- Defer any abstraction that does not help the MVP work.
- Keep Python as the orchestration and user-facing layer.
- Keep MLIR as the real compiler backbone, not as a decorative intermediate.

## Open Technical Choices To Resolve Early

These decisions should be settled quickly in Phase 1 or early Phase 2:

- Whether the first MLIR generation path is pure Python text emission or a small C++ helper tool.
- Whether runtime buffer support is built around `cuda-python`, another CUDA binding, or a very small custom bridge.
- Whether kernel MLIR should begin from standard dialects only or include a tiny custom dialect immediately.
- How compute capability is selected for PTX generation and cached artifacts.

The default bias should be toward the simplest option that keeps the MLIR path real and the runtime launch path debuggable.

## Next Implementation Step

The next step should be Phase 1 execution only:

- create the minimal repository scaffolding,
- add build/configuration files that point to the local LLVM/MLIR build,
- and record the exact MVP compile strategy decision,

but no source code for the compiler pipeline should be written until that scaffolding is in place.

## Phase 1 Status

Phase 1 scaffolding is now the intended implementation target:

- top-level project metadata in `pyproject.toml`,
- local LLVM/MLIR path configuration in `CMakeLists.txt` and `cmake/LLVMConfigHelpers.cmake`,
- a minimal Python package stub in `python/mini_triton/__init__.py`,
- and the MVP compile strategy decision record in `docs/decisions/0001-mvp-compile-strategy.md`.

The next coding phase after this scaffold is accepted should begin with the Python kernel declaration and symbolic capture path for vector add, not with a larger compiler subsystem.