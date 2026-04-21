from __future__ import annotations

from dataclasses import dataclass
from inspect import Signature, signature
from pathlib import Path
from typing import Any, Callable

from .cache import CachedCompilation, compile_with_cache
from .compiler import CompilationPlan, CompilationRun, create_compilation_plan, execute_compilation_plan
from .capture import BufferProxy, CaptureError, CaptureSession, KernelContextProxy, ScalarProxy
from .ir import ArgSpec, KernelIR
from .mlir import emit_mlir


def buffer(dtype: str = "float32", address_space: str = "global") -> ArgSpec:
    return ArgSpec(kind="buffer", dtype=dtype, address_space=address_space)


def scalar(dtype: str = "int32") -> ArgSpec:
    return ArgSpec(kind="scalar", dtype=dtype)


@dataclass(frozen=True)
class Kernel:
    fn: Callable[..., Any]
    block_size: int
    signature: Signature

    def capture(self, **arg_specs: ArgSpec) -> KernelIR:
        parameter_names = list(self.signature.parameters.keys())
        if not parameter_names or parameter_names[0] != "ctx":
            raise CaptureError("kernel functions must declare 'ctx' as the first parameter")

        expected_names = parameter_names[1:]
        provided_names = list(arg_specs.keys())

        if expected_names != provided_names:
            raise CaptureError(
                "kernel.capture argument specs must match the declared parameter order: "
                f"expected {expected_names}, got {provided_names}"
            )

        session = CaptureSession(self.fn.__name__, self.block_size, arg_specs)
        runtime_args = [KernelContextProxy(session)]

        for name in expected_names:
            spec = arg_specs[name]
            if spec.kind == "buffer":
                runtime_args.append(BufferProxy(session, name, spec))
            elif spec.kind == "scalar":
                runtime_args.append(ScalarProxy(session, name, spec.dtype))
            else:
                raise CaptureError(f"unsupported argument kind: {spec.kind}")

        result = self.fn(*runtime_args)
        if result is not None:
            raise CaptureError("kernel functions must not return a value during capture")

        if not session.ir.stores:
            raise CaptureError("kernel capture produced no stores; expected at least one output write")

        return session.ir

    def emit_mlir(self, **arg_specs: ArgSpec) -> str:
        return emit_mlir(self.capture(**arg_specs))

    def plan_compile(
        self,
        output_dir: str | Path,
        *,
        llvm_build_dir: str | Path | None = None,
        cpp_lowering_driver: str | Path | None = None,
        cuda_arch: str = "sm_80",
        **arg_specs: ArgSpec,
    ) -> CompilationPlan:
        return create_compilation_plan(
            self.capture(**arg_specs),
            output_dir=output_dir,
            llvm_build_dir=llvm_build_dir,
            cpp_lowering_driver=cpp_lowering_driver,
            cuda_arch=cuda_arch,
        )

    def compile(
        self,
        output_dir: str | Path,
        *,
        llvm_build_dir: str | Path | None = None,
        cpp_lowering_driver: str | Path | None = None,
        cuda_arch: str = "sm_80",
        **arg_specs: ArgSpec,
    ) -> CompilationRun:
        plan = self.plan_compile(
            output_dir,
            llvm_build_dir=llvm_build_dir,
            cpp_lowering_driver=cpp_lowering_driver,
            cuda_arch=cuda_arch,
            **arg_specs,
        )
        return execute_compilation_plan(plan)

    def compile_cached(
        self,
        output_dir: str | Path,
        *,
        llvm_build_dir: str | Path | None = None,
        cpp_lowering_driver: str | Path | None = None,
        cuda_arch: str = "sm_80",
        **arg_specs: ArgSpec,
    ) -> CachedCompilation:
        return compile_with_cache(
            self.capture(**arg_specs),
            output_dir=output_dir,
            llvm_build_dir=llvm_build_dir,
            cpp_lowering_driver=cpp_lowering_driver,
            cuda_arch=cuda_arch,
        )


def kernel(*, block_size: int) -> Callable[[Callable[..., Any]], Kernel]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    def decorate(fn: Callable[..., Any]) -> Kernel:
        return Kernel(fn=fn, block_size=block_size, signature=signature(fn))

    return decorate