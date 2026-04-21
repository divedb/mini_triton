# Decision 0001: MVP Compile Strategy

## Status

Accepted.

## Context

The project needs a real MLIR-to-CUDA compilation path early, but it should avoid unnecessary implementation weight before the first end-to-end kernel works. The repository already assumes that a local LLVM/MLIR build exists elsewhere on disk, so Phase 1 should take advantage of that fact rather than introducing a custom compiler binary immediately.

## Decision

The MVP compile path will be orchestrated primarily from Python and will invoke the existing local LLVM/MLIR command-line tools directly.

The initial route is:

1. Python capture produces a tiny internal kernel IR.
2. Python lowers that IR into MLIR text.
3. Python invokes local `mlir-opt` if structural cleanup or verification is needed.
4. Python invokes local `mlir-translate` to produce LLVM IR.
5. Python invokes local `llc` targeting NVPTX to produce PTX.
6. Python runtime code loads PTX and launches the kernel through a CUDA binding.

## Consequences

Positive:

- Keeps Phase 2 through Phase 5 focused on one vertical slice.
- Reuses the existing local LLVM/MLIR build exactly as intended.
- Makes compiler artifacts easy to inspect because every stage is a file or subprocess boundary.
- Avoids committing to a custom C++ MLIR driver before there is evidence that one is necessary.

Negative:

- Tool orchestration from Python is slower than an in-process compiler path.
- Diagnostics may initially be split across subprocess boundaries.
- The pipeline will need careful artifact and error handling.

## Deferred Alternative

A small in-repo C++ compiler driver remains a valid later optimization if one of these becomes true:

- subprocess overhead becomes a measurable problem,
- MLIR pass construction becomes awkward from the Python-side orchestration,
- or tighter in-process diagnostics are needed.

That alternative is explicitly deferred until after the first end-to-end kernel works.