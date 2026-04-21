from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .dialect import emit_mini_dialect
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
    cpp_lowering_driver: Path | None = None


@dataclass(frozen=True)
class ArtifactPaths:
    root_dir: Path
    input_mlir: Path
    input_dialect: Path | None
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
    dialect_text: str | None
    artifacts: ArtifactPaths
    toolchain: ToolchainPaths
    commands: tuple[CompilationCommand, ...]

    def materialize(self) -> ArtifactPaths:
        self.artifacts.root_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.input_mlir.write_text(self.module_text + "\n", encoding="utf-8")
        if self.artifacts.input_dialect is not None and self.dialect_text is not None:
            self.artifacts.input_dialect.write_text(self.dialect_text + "\n", encoding="utf-8")
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


def toolchain_fingerprint(toolchain: ToolchainPaths) -> str:
    records = [
        _tool_fingerprint_record("mlir-opt", toolchain.mlir_opt),
        _tool_fingerprint_record("mlir-translate", toolchain.mlir_translate),
        _tool_fingerprint_record("llc", toolchain.llc),
    ]
    if toolchain.cpp_lowering_driver is not None:
        records.append(_tool_fingerprint_record("cpp-lowering-driver", toolchain.cpp_lowering_driver))
    return "|".join(records)


def _tool_fingerprint_record(tool_name: str, path: Path) -> str:
    stats = path.stat()
    return f"{tool_name}={path}:{stats.st_size}:{int(stats.st_mtime_ns)}"


def _resolve_tool_executable(bin_dir: Path, tool_name: str) -> Path | None:
    direct = bin_dir / tool_name
    if direct.is_file():
        return direct

    if os.name == "nt":
        for extension in (".exe", ".cmd", ".bat"):
            candidate = bin_dir / f"{tool_name}{extension}"
            if candidate.is_file():
                return candidate

    return None


def _find_llvm_bin_dir(build_dir: Path) -> Path | None:
    # Prefer direct bin/ first, then common multi-config layouts.
    candidates = [
        build_dir / "bin",
        build_dir / "Release" / "bin",
        build_dir / "RelWithDebInfo" / "bin",
        build_dir / "Debug" / "bin",
        build_dir / "MinSizeRel" / "bin",
        build_dir / "x64" / "Release" / "bin",
        build_dir / "x64" / "RelWithDebInfo" / "bin",
        build_dir / "x64" / "Debug" / "bin",
        build_dir / "x64" / "MinSizeRel" / "bin",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def resolve_toolchain(
    llvm_build_dir: str | os.PathLike[str] | None = None,
    *,
    cpp_lowering_driver: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ToolchainPaths:
    environment = env if env is not None else os.environ
    configured_build_dir = llvm_build_dir or environment.get("MINITRITON_LLVM_BUILD_DIR")
    if not configured_build_dir:
        raise CompilerConfigError(
            "LLVM build directory is required; pass llvm_build_dir or set MINITRITON_LLVM_BUILD_DIR"
        )

    build_dir = Path(configured_build_dir).expanduser().resolve()
    bin_dir = _find_llvm_bin_dir(build_dir)
    if bin_dir is None:
        raise CompilerConfigError(
            "LLVM build directory does not contain a recognizable tool bin/ path: "
            f"{build_dir} (expected e.g. bin/ or Release/bin)"
        )

    tool_paths = {
        "mlir-opt": _resolve_tool_executable(bin_dir, "mlir-opt"),
        "mlir-translate": _resolve_tool_executable(bin_dir, "mlir-translate"),
        "llc": _resolve_tool_executable(bin_dir, "llc"),
    }
    missing_tools = [tool_name for tool_name, tool_path in tool_paths.items() if tool_path is None]
    if missing_tools:
        joined = ", ".join(missing_tools)
        raise CompilerConfigError(f"LLVM toolchain is missing required tools under {bin_dir}: {joined}")

    _validate_llc_nvptx_support(tool_paths["llc"])

    resolved_cpp_driver = _resolve_cpp_lowering_driver(
        cpp_lowering_driver,
        environment,
        llvm_build_dir=build_dir,
    )

    return ToolchainPaths(
        llvm_build_dir=build_dir,
        mlir_opt=tool_paths["mlir-opt"],
        mlir_translate=tool_paths["mlir-translate"],
        llc=tool_paths["llc"],
        cpp_lowering_driver=resolved_cpp_driver,
    )


def _resolve_cpp_lowering_driver(
    configured_driver: str | os.PathLike[str] | None,
    environment: Mapping[str, str],
    *,
    llvm_build_dir: Path,
) -> Path | None:
    candidate = configured_driver or environment.get("MINITRITON_CPP_LOWERING_DRIVER")
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise CompilerConfigError(f"C++ lowering driver path does not exist: {path}")
        return path

    return _auto_detect_cpp_lowering_driver(llvm_build_dir)


def _auto_detect_cpp_lowering_driver(llvm_build_dir: Path) -> Path | None:
    repo_root = _infer_repo_root_from_llvm_build_dir(llvm_build_dir)
    if repo_root is None:
        return None

    candidates = [
        repo_root / "build" / "host" / "Release" / "mini_triton_lower.exe",
        repo_root / "build" / "host" / "RelWithDebInfo" / "mini_triton_lower.exe",
        repo_root / "build" / "host" / "Debug" / "mini_triton_lower.exe",
        repo_root / "build" / "host" / "MinSizeRel" / "mini_triton_lower.exe",
        repo_root / "build" / "host" / "mini_triton_lower.exe",
        repo_root / "build" / "host" / "mini_triton_lower",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _infer_repo_root_from_llvm_build_dir(llvm_build_dir: Path) -> Path | None:
    normalized = llvm_build_dir.resolve()
    markers = ("tools/mtc-lower/main.cpp", "python/mini_triton/__init__.py", "CMakeLists.txt")

    for ancestor in (normalized,) + tuple(normalized.parents):
        if all((ancestor / marker).is_file() for marker in markers):
            return ancestor

    return None


def _validate_llc_nvptx_support(llc_path: Path) -> None:
    """Best-effort validation that llc was built with NVPTX backend support.

    If llc does not support --version in a test double, this check is skipped.
    """

    supported, _targets = inspect_llc_nvptx_support(llc_path)
    if supported is not False:
        return

    raise CompilerConfigError(
        "Configured llc does not include NVPTX backend support. "
        "Rebuild LLVM with NVPTX enabled (for example: -DLLVM_TARGETS_TO_BUILD=NVPTX;X86)."
    )


def inspect_llc_nvptx_support(llc_path: Path) -> tuple[bool | None, list[str] | None]:
    """Return NVPTX support status and parsed llc registered targets.

    Returns:
      - (True, targets): NVPTX is present in parsed targets
      - (False, targets): targets were parsed and NVPTX is missing
      - (None, None): could not determine support from llc --version output
    """

    try:
        completed = subprocess.run(
            [str(llc_path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None

    if completed.returncode != 0:
        return None, None

    targets = _parse_llc_registered_targets(completed.stdout or "")
    if targets is None:
        return None, None

    supports_nvptx = any("nvptx" in target.lower() for target in targets)
    return supports_nvptx, targets


def _parse_llc_registered_targets(version_text: str) -> list[str] | None:
    lines = version_text.splitlines()
    start_index = None

    for index, line in enumerate(lines):
        if line.strip().lower().startswith("registered targets"):
            start_index = index + 1
            break

    if start_index is None:
        return None

    targets: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped and not stripped[0].isalnum():
            # Defensive guard for oddly formatted continuation headers.
            continue

        token = stripped.split(maxsplit=1)[0]
        if token:
            targets.append(token)

    return targets


def create_compilation_plan(
    kernel_ir: KernelIR,
    *,
    output_dir: str | os.PathLike[str],
    llvm_build_dir: str | os.PathLike[str] | None = None,
    cpp_lowering_driver: str | os.PathLike[str] | None = None,
    cuda_arch: str = "sm_80",
) -> CompilationPlan:
    toolchain = resolve_toolchain(llvm_build_dir, cpp_lowering_driver=cpp_lowering_driver)
    root_dir = Path(output_dir).expanduser().resolve() / kernel_ir.name
    artifacts = ArtifactPaths(
        root_dir=root_dir,
        input_mlir=root_dir / f"{kernel_ir.name}.mlir",
        input_dialect=(root_dir / f"{kernel_ir.name}.mtd") if toolchain.cpp_lowering_driver is not None else None,
        optimized_mlir=root_dir / f"{kernel_ir.name}.opt.mlir",
        llvm_ir=root_dir / f"{kernel_ir.name}.ll",
        ptx=root_dir / f"{kernel_ir.name}.ptx",
    )

    dialect_text = emit_mini_dialect(kernel_ir) if toolchain.cpp_lowering_driver is not None else None

    if toolchain.cpp_lowering_driver is not None:
        commands = (
            CompilationCommand(
                name="cpp-lowering-driver",
                argv=(
                    str(toolchain.cpp_lowering_driver),
                    "--mlir-opt",
                    str(toolchain.mlir_opt),
                    "--mlir-translate",
                    str(toolchain.mlir_translate),
                    "--llc",
                    str(toolchain.llc),
                    "--input-dialect",
                    str(artifacts.input_dialect),
                    "--input-mlir",
                    str(artifacts.input_mlir),
                    "--optimized-mlir",
                    str(artifacts.optimized_mlir),
                    "--llvm-ir",
                    str(artifacts.llvm_ir),
                    "--ptx",
                    str(artifacts.ptx),
                    "--cuda-arch",
                    cuda_arch,
                ),
                output_path=artifacts.ptx,
            ),
        )
    else:
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
        dialect_text=dialect_text,
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