from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mini_triton import CompilerConfigError, buffer, kernel, resolve_toolchain, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


@kernel(block_size=64)
def add_kernel_with_arange(ctx, x, y, out, n):
    idx = ctx.arange(1, 1025)
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


@kernel(block_size=64)
def add_kernel_with_arange_step(ctx, x, y, out, n):
    idx = ctx.arange(0, 2048, 2)
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


def test_resolve_toolchain_uses_existing_llvm_build_tree(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    toolchain = resolve_toolchain(build_dir)

    assert toolchain.llvm_build_dir == build_dir.resolve()
    assert toolchain.mlir_opt.name in {"mlir-opt", "mlir-opt.exe", "mlir-opt.cmd", "mlir-opt.bat"}
    assert toolchain.mlir_translate.name in {
        "mlir-translate",
        "mlir-translate.exe",
        "mlir-translate.cmd",
        "mlir-translate.bat",
    }
    assert toolchain.llc.name in {"llc", "llc.exe", "llc.cmd", "llc.bat"}


def test_resolve_toolchain_finds_tools_under_release_bin(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    release_bin_dir = build_dir / "Release" / "bin"
    release_bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(release_bin_dir, tool_name, "exit /b 0\n")

    toolchain = resolve_toolchain(build_dir)

    assert toolchain.llvm_build_dir == build_dir.resolve()
    assert toolchain.mlir_opt.parent == release_bin_dir.resolve()
    assert toolchain.mlir_translate.parent == release_bin_dir.resolve()
    assert toolchain.llc.parent == release_bin_dir.resolve()


def test_resolve_toolchain_requires_expected_tools(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    (build_dir / "bin").mkdir(parents=True)
    _write_tool(build_dir / "bin", "mlir-opt", "exit /b 0\n")

    try:
        resolve_toolchain(build_dir)
    except CompilerConfigError as exc:
        assert "mlir-translate" in str(exc)
        assert "llc" in str(exc)
    else:
        raise AssertionError("expected toolchain resolution to fail when tools are missing")


def test_resolve_toolchain_rejects_llc_without_nvptx(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    _write_tool(bin_dir, "mlir-opt", "exit /b 0\n")
    _write_tool(bin_dir, "mlir-translate", "exit /b 0\n")

    if os.name == "nt":
        _write_tool(
            bin_dir,
            "llc",
            "if \"%~1\"==\"--version\" (\n"
            "  echo LLVM fake\n"
            "  echo Registered Targets:\n"
            "  echo   x86\n"
            "  exit /b 0\n"
            ")\n"
            "exit /b 0\n",
        )
    else:
        _write_tool(
            bin_dir,
            "llc",
            "if [ \"${1:-}\" = \"--version\" ]; then\n"
            "  echo 'LLVM fake'\n"
            "  echo 'Registered Targets:'\n"
            "  echo '  x86'\n"
            "  exit 0\n"
            "fi\n"
            "exit 0\n",
        )

    with pytest.raises(CompilerConfigError, match="NVPTX"):
        resolve_toolchain(build_dir)


def test_plan_compile_materializes_input_mlir_and_commands(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)
    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    plan = add_kernel.plan_compile(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    artifacts = plan.materialize()

    assert artifacts.input_mlir.exists()
    assert artifacts.input_mlir.read_text(encoding="utf-8").startswith("module {\n")
    assert artifacts.optimized_mlir.name == "add_kernel.opt.mlir"
    assert artifacts.llvm_ir.name == "add_kernel.ll"
    assert artifacts.ptx.name == "add_kernel.ptx"

    assert plan.commands[0].name == "mlir-opt"
    assert plan.commands[0].argv[1:] == (
        str(artifacts.input_mlir),
        "--convert-nvvm-to-llvm",
        "--reconcile-unrealized-casts",
        "-o",
        str(artifacts.optimized_mlir),
    )
    assert Path(plan.commands[0].argv[0]).name in {
        "mlir-opt",
        "mlir-opt.exe",
        "mlir-opt.cmd",
        "mlir-opt.bat",
    }
    assert plan.commands[1].argv[1] == "--mlir-to-llvmir"
    assert "-mcpu=sm_80" in plan.commands[2].argv


def test_plan_compile_uses_cpp_lowering_driver_when_configured(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    cpp_driver = tmp_path / ("mini_triton_lower.cmd" if os.name == "nt" else "mini_triton_lower")
    if os.name == "nt":
        cpp_driver.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
    else:
        cpp_driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        _make_executable(cpp_driver)

    plan = add_kernel.plan_compile(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        cpp_lowering_driver=cpp_driver,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert len(plan.commands) == 1
    assert plan.commands[0].name == "cpp-lowering-driver"
    assert plan.commands[0].argv[0] == str(cpp_driver.resolve())
    assert "--input-dialect" in plan.commands[0].argv
    assert "--cuda-arch" in plan.commands[0].argv

    artifacts = plan.materialize()
    assert artifacts.input_dialect is not None
    assert artifacts.input_dialect.exists()
    assert artifacts.input_dialect.read_text(encoding="utf-8").startswith("kernel add_kernel 128\n")


def test_resolve_toolchain_auto_detects_cpp_lowering_driver(tmp_path: Path):
    repo_root = tmp_path / "repo"
    build_dir = repo_root / "third_party" / "llvm-project" / "build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    (repo_root / "tools" / "mtc-lower").mkdir(parents=True)
    (repo_root / "tools" / "mtc-lower" / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (repo_root / "python" / "mini_triton").mkdir(parents=True)
    (repo_root / "python" / "mini_triton" / "__init__.py").write_text("\n", encoding="utf-8")
    (repo_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")

    cpp_driver = repo_root / "build" / "host" / "Release" / (
        "mini_triton_lower.exe" if os.name == "nt" else "mini_triton_lower"
    )
    cpp_driver.parent.mkdir(parents=True)
    cpp_driver.write_text("@echo off\nexit /b 0\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n", encoding="utf-8")
    _make_executable(cpp_driver)

    toolchain = resolve_toolchain(build_dir)
    assert toolchain.cpp_lowering_driver == cpp_driver.resolve()


def test_cpp_lowering_plan_materializes_arange_dialect(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    cpp_driver = tmp_path / ("mini_triton_lower.cmd" if os.name == "nt" else "mini_triton_lower")
    if os.name == "nt":
        cpp_driver.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
    else:
        cpp_driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        _make_executable(cpp_driver)

    plan = add_kernel_with_arange.plan_compile(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        cpp_lowering_driver=cpp_driver,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    artifacts = plan.materialize()
    assert artifacts.input_dialect is not None
    dialect_text = artifacts.input_dialect.read_text(encoding="utf-8")
    assert "value v0 arange index" in dialect_text


def test_cpp_lowering_plan_materializes_arange_step_dialect(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_tool(bin_dir, tool_name, "exit /b 0\n")

    cpp_driver = tmp_path / ("mini_triton_lower.cmd" if os.name == "nt" else "mini_triton_lower")
    if os.name == "nt":
        cpp_driver.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
    else:
        cpp_driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        _make_executable(cpp_driver)

    plan = add_kernel_with_arange_step.plan_compile(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        cpp_lowering_driver=cpp_driver,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    artifacts = plan.materialize()
    assert artifacts.input_dialect is not None
    dialect_text = artifacts.input_dialect.read_text(encoding="utf-8")
    assert "step=2" in dialect_text


def test_compile_executes_planned_toolchain_commands(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_copy_tool(bin_dir, tool_name)

    run = add_kernel.compile(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert len(run.results) == 3
    assert all(result.returncode == 0 for result in run.results)
    assert run.plan.artifacts.optimized_mlir.exists()
    assert run.plan.artifacts.llvm_ir.exists()
    assert run.plan.artifacts.ptx.exists()


def test_compile_with_real_toolchain_emits_ptx_kernel_entry(tmp_path: Path):
    llvm_build_dir = os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if not llvm_build_dir:
        pytest.skip("MINITRITON_LLVM_BUILD_DIR is not configured")

    try:
        resolve_toolchain(llvm_build_dir)
    except CompilerConfigError as exc:
        pytest.skip(f"configured LLVM toolchain is not usable: {exc}")

    run = add_kernel.compile(
        tmp_path / "artifacts",
        llvm_build_dir=llvm_build_dir,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert [result.returncode for result in run.results] == [0, 0, 0]
    assert run.plan.artifacts.input_mlir.exists()
    assert run.plan.artifacts.optimized_mlir.exists()
    assert run.plan.artifacts.llvm_ir.exists()
    assert run.plan.artifacts.ptx.exists()

    ptx_text = run.plan.artifacts.ptx.read_text(encoding="utf-8")
    assert ".entry add_kernel(" in ptx_text


def _make_executable(path: Path) -> None:
    if os.name == "nt":
        return
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR)


def _write_tool(bin_dir: Path, tool_name: str, body: str) -> None:
    if os.name == "nt":
        path = bin_dir / f"{tool_name}.cmd"
        path.write_text("@echo off\n" + body, encoding="utf-8")
        return

    path = bin_dir / tool_name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    _make_executable(path)


def _write_copy_tool(bin_dir: Path, tool_name: str) -> None:
    if os.name == "nt":
        _write_tool(
            bin_dir,
            tool_name,
            "setlocal EnableDelayedExpansion\n"
            "set \"input=\"\n"
            "set \"output=\"\n"
            "set \"prev=\"\n"
            ":loop\n"
            "if \"%~1\"==\"\" goto done\n"
            "if \"%prev%\"==\"-o\" (\n"
            "  set \"output=%~1\"\n"
            "  set \"prev=\"\n"
            "  shift\n"
            "  goto loop\n"
            ")\n"
            "if \"%~1\"==\"-o\" (\n"
            "  set \"prev=-o\"\n"
            "  shift\n"
            "  goto loop\n"
            ")\n"
            "set \"arg=%~1\"\n"
            "if not \"!arg:~0,1!\"==\"-\" if \"%input%\"==\"\" set \"input=%~1\"\n"
            "shift\n"
            "goto loop\n"
            ":done\n"
            "if \"%output%\"==\"\" exit /b 1\n"
            "if not \"%input%\"==\"\" if exist \"%input%\" (\n"
            "  copy /Y \"%input%\" \"%output%\" >nul\n"
            ") else (\n"
            "  type nul > \"%output%\"\n"
            ")\n"
            "endlocal\n",
        )
        return

    _write_tool(
        bin_dir,
        tool_name,
        "set -eu\n"
        "input=''\n"
        "output=''\n"
        "prev=''\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = '-o' ]; then\n"
        "    output=\"$arg\"\n"
        "    prev=''\n"
        "    continue\n"
        "  fi\n"
        "  if [ \"$arg\" = '-o' ]; then\n"
        "    prev='-o'\n"
        "    continue\n"
        "  fi\n"
        "  case \"$arg\" in\n"
        "    -*) continue ;;\n"
        "  esac\n"
        "  if [ -z \"$input\" ]; then\n"
        "    input=\"$arg\"\n"
        "  fi\n"
        "done\n"
        "if [ -z \"$output\" ]; then\n"
        "  exit 1\n"
        "fi\n"
        "if [ -n \"$input\" ] && [ -f \"$input\" ]; then\n"
        "  cp \"$input\" \"$output\"\n"
        "else\n"
        "  : > \"$output\"\n"
        "fi\n",
    )