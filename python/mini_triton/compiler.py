from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .ir import KernelIR
from .mlir import emit_mlir


class CompilerConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolchainPaths:
    llvm_build_dir: Path
    mlir_opt: Path
    mlir_translate: Path
    llc: Path


@dataclass(frozen=True)
class ArtifactPaths:
    root_dir: Path
    input_mlir: Path
    optimized_mlir: Path
    llvm_ir: Path
    ptx: Path


@dataclass(frozen=True)
class CompilationCommand:
    name: str
    argv: tuple[str, ...]
    output_path: Path


@dataclass(frozen=True)
class CompilationPlan:
    kernel_name: str
    module_text: str
    artifacts: ArtifactPaths
    toolchain: ToolchainPaths
    commands: tuple[CompilationCommand, ...]

    def materialize(self) -> ArtifactPaths:
        self.artifacts.root_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.input_mlir.write_text(self.module_text + "\n", encoding="utf-8")
        return self.artifacts


@dataclass(frozen=True)
class CommandExecutionResult:
    command: CompilationCommand
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CompilationRun:
    plan: CompilationPlan
    results: tuple[CommandExecutionResult, ...]


def resolve_toolchain(
    llvm_build_dir: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ToolchainPaths:
    environment = env if env is not None else os.environ
    configured_build_dir = llvm_build_dir or environment.get("MINITRITON_LLVM_BUILD_DIR")
    if not configured_build_dir:
        raise CompilerConfigError(
            "LLVM build directory is required; pass llvm_build_dir or set MINITRITON_LLVM_BUILD_DIR"
        )

    build_dir = Path(configured_build_dir).expanduser().resolve()
    bin_dir = build_dir / "bin"
    if not bin_dir.is_dir():
        raise CompilerConfigError(f"LLVM build directory does not contain bin/: {build_dir}")

    tool_paths = {
        "mlir-opt": bin_dir / "mlir-opt",
        "mlir-translate": bin_dir / "mlir-translate",
        "llc": bin_dir / "llc",
    }
    missing_tools = [tool_name for tool_name, tool_path in tool_paths.items() if not tool_path.is_file()]
    if missing_tools:
        joined = ", ".join(missing_tools)
        raise CompilerConfigError(f"LLVM toolchain is missing required tools under {bin_dir}: {joined}")

    return ToolchainPaths(
        llvm_build_dir=build_dir,
        mlir_opt=tool_paths["mlir-opt"],
        mlir_translate=tool_paths["mlir-translate"],
        llc=tool_paths["llc"],
    )


def create_compilation_plan(
    kernel_ir: KernelIR,
    *,
    output_dir: str | os.PathLike[str],
    llvm_build_dir: str | os.PathLike[str] | None = None,
    cuda_arch: str = "sm_80",
) -> CompilationPlan:
    toolchain = resolve_toolchain(llvm_build_dir)
    root_dir = Path(output_dir).expanduser().resolve() / kernel_ir.name
    artifacts = ArtifactPaths(
        root_dir=root_dir,
        input_mlir=root_dir / f"{kernel_ir.name}.mlir",
        optimized_mlir=root_dir / f"{kernel_ir.name}.opt.mlir",
        llvm_ir=root_dir / f"{kernel_ir.name}.ll",
        ptx=root_dir / f"{kernel_ir.name}.ptx",
    )

    commands = (
        CompilationCommand(
            name="mlir-opt",
            argv=(
                str(toolchain.mlir_opt),
                str(artifacts.input_mlir),
                "--convert-nvvm-to-llvm",
                "--reconcile-unrealized-casts",
                "-o",
                str(artifacts.optimized_mlir),
            ),
            output_path=artifacts.optimized_mlir,
        ),
        CompilationCommand(
            name="mlir-translate",
            argv=(
                str(toolchain.mlir_translate),
                "--mlir-to-llvmir",
                str(artifacts.optimized_mlir),
                "-o",
                str(artifacts.llvm_ir),
            ),
            output_path=artifacts.llvm_ir,
        ),
        CompilationCommand(
            name="llc",
            argv=(
                str(toolchain.llc),
                "-march=nvptx64",
                f"-mcpu={cuda_arch}",
                str(artifacts.llvm_ir),
                "-o",
                str(artifacts.ptx),
            ),
            output_path=artifacts.ptx,
        ),
    )

    return CompilationPlan(
        kernel_name=kernel_ir.name,
        module_text=emit_mlir(kernel_ir),
        artifacts=artifacts,
        toolchain=toolchain,
        commands=commands,
    )


def execute_compilation_plan(
    plan: CompilationPlan,
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> CompilationRun:
    plan.materialize()

    environment = None if env is None else dict(env)
    working_dir = None if cwd is None else Path(cwd)
    results: list[CommandExecutionResult] = []

    for command in plan.commands:
        completed = subprocess.run(
            command.argv,
            cwd=working_dir,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            CommandExecutionResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command.argv,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    return CompilationRun(plan=plan, results=tuple(results))