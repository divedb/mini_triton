from __future__ import annotations

from pathlib import Path

from mini_triton import buffer, kernel, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


def test_compile_cached_reuses_existing_ptx(tmp_path: Path):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_copy_tool(bin_dir, tool_name)

    first = add_kernel.compile_cached(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    second = add_kernel.compile_cached(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert first.cache_hit is False
    assert first.run is not None
    assert second.cache_hit is True
    assert second.run is None
    assert first.ptx_path == second.ptx_path
    assert second.ptx_path.exists()


def test_compile_cached_invalidates_when_toolchain_changes(tmp_path: Path):
    build_dir_a = tmp_path / "llvm-build-a"
    build_dir_b = tmp_path / "llvm-build-b"
    for build_dir in (build_dir_a, build_dir_b):
        bin_dir = build_dir / "bin"
        bin_dir.mkdir(parents=True)
        for tool_name in ("mlir-opt", "mlir-translate", "llc"):
            _write_copy_tool(bin_dir, tool_name)

    first = add_kernel.compile_cached(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir_a,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    second = add_kernel.compile_cached(
        tmp_path / "artifacts",
        llvm_build_dir=build_dir_b,
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert first.cache_key != second.cache_key
    assert first.cache_hit is False
    assert second.cache_hit is False


def _write_tool(bin_dir: Path, tool_name: str, body: str) -> None:
    import os
    import stat

    if os.name == "nt":
        path = bin_dir / f"{tool_name}.cmd"
        path.write_text("@echo off\n" + body, encoding="utf-8")
        return

    path = bin_dir / tool_name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR)


def _write_copy_tool(bin_dir: Path, tool_name: str) -> None:
    import os

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
