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


def test_resolve_toolchain_uses_existing_llvm_build_tree(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        tool_path = bin_dir / tool_name
        tool_path.write_text("#!/bin/sh\n", encoding="utf-8")
        _make_executable(tool_path)

    toolchain = resolve_toolchain(build_dir)

    assert toolchain.llvm_build_dir == build_dir.resolve()
    assert toolchain.mlir_opt == (bin_dir / "mlir-opt").resolve()
    assert toolchain.mlir_translate == (bin_dir / "mlir-translate").resolve()
    assert toolchain.llc == (bin_dir / "llc").resolve()


def test_resolve_toolchain_requires_expected_tools(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    (build_dir / "bin").mkdir(parents=True)
    (build_dir / "bin" / "mlir-opt").write_text("#!/bin/sh\n", encoding="utf-8")

    try:
        resolve_toolchain(build_dir)
    except CompilerConfigError as exc:
        assert "mlir-translate" in str(exc)
        assert "llc" in str(exc)
    else:
        raise AssertionError("expected toolchain resolution to fail when tools are missing")


def test_plan_compile_materializes_input_mlir_and_commands(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)
    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        tool_path = bin_dir / tool_name
        tool_path.write_text("#!/bin/sh\n", encoding="utf-8")
        _make_executable(tool_path)

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
    assert plan.commands[0].argv == (
        str((bin_dir / "mlir-opt").resolve()),
        str(artifacts.input_mlir),
        "--convert-nvvm-to-llvm",
        "--reconcile-unrealized-casts",
        "-o",
        str(artifacts.optimized_mlir),
    )
    assert plan.commands[1].argv[1] == "--mlir-to-llvmir"
    assert "-mcpu=sm_80" in plan.commands[2].argv


def test_compile_executes_planned_toolchain_commands(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        tool_path = bin_dir / tool_name
        tool_path.write_text(
            "#!/bin/sh\n"
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
            "  echo 'missing output' >&2\n"
            "  exit 1\n"
            "fi\n"
            "if [ -n \"$input\" ] && [ -f \"$input\" ]; then\n"
            "  cp \"$input\" \"$output\"\n"
            "else\n"
            "  : > \"$output\"\n"
            "fi\n",
            encoding="utf-8",
        )
        _make_executable(tool_path)

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
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR)