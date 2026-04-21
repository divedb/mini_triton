from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .compiler import (
    CompilationRun,
    create_compilation_plan,
    execute_compilation_plan,
    resolve_toolchain,
    toolchain_fingerprint,
)
from .ir import KernelIR


@dataclass(frozen=True)
class CachedCompilation:
    cache_key: str
    cache_root: Path
    ptx_path: Path
    cache_hit: bool
    run: CompilationRun | None


def compile_with_cache(
    kernel_ir: KernelIR,
    *,
    output_dir: str | Path,
    llvm_build_dir: str | Path | None = None,
    cpp_lowering_driver: str | Path | None = None,
    cuda_arch: str = "sm_80",
) -> CachedCompilation:
    resolved_toolchain = resolve_toolchain(llvm_build_dir, cpp_lowering_driver=cpp_lowering_driver)
    toolchain_tag = toolchain_fingerprint(resolved_toolchain)
    cache_key = _cache_key(kernel_ir, cuda_arch, toolchain_tag)
    cache_root = Path(output_dir).expanduser().resolve() / "cache" / cache_key
    ptx_path = cache_root / kernel_ir.name / f"{kernel_ir.name}.ptx"

    if ptx_path.is_file():
        return CachedCompilation(
            cache_key=cache_key,
            cache_root=cache_root,
            ptx_path=ptx_path,
            cache_hit=True,
            run=None,
        )

    plan = create_compilation_plan(
        kernel_ir,
        output_dir=cache_root,
        llvm_build_dir=resolved_toolchain.llvm_build_dir,
        cpp_lowering_driver=resolved_toolchain.cpp_lowering_driver,
        cuda_arch=cuda_arch,
    )
    run = execute_compilation_plan(plan)

    _write_cache_metadata(
        cache_root,
        kernel_ir,
        cuda_arch,
        run.plan.artifacts.ptx,
        toolchain_tag,
    )
    return CachedCompilation(
        cache_key=cache_key,
        cache_root=cache_root,
        ptx_path=run.plan.artifacts.ptx,
        cache_hit=False,
        run=run,
    )


def _cache_key(kernel_ir: KernelIR, cuda_arch: str, toolchain_tag: str) -> str:
    payload = "\n".join(
        [
            f"kernel={kernel_ir.name}",
            f"block_size={kernel_ir.block_size}",
            f"cuda_arch={cuda_arch}",
            f"toolchain={toolchain_tag}",
            kernel_ir.format(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write_cache_metadata(
    cache_root: Path,
    kernel_ir: KernelIR,
    cuda_arch: str,
    ptx_path: Path,
    toolchain_tag: str,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kernel": kernel_ir.name,
        "block_size": kernel_ir.block_size,
        "cuda_arch": cuda_arch,
        "toolchain": toolchain_tag,
        "ptx_path": str(ptx_path),
    }
    (cache_root / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
