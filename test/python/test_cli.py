from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from mini_triton.cli import main


def test_cli_compile_add_with_cache(tmp_path: Path, capsys):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    for tool_name in ("mlir-opt", "mlir-translate", "llc"):
        _write_copy_tool(bin_dir, tool_name)

    output_dir = tmp_path / "artifacts"
    rc = main(
        [
            "compile-add",
            "--llvm-build-dir",
            str(build_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kernel"] == "add_kernel"
    assert payload["cache_hit"] is False
    assert Path(payload["ptx_path"]).exists()

    rc2 = main(
        [
            "compile-add",
            "--llvm-build-dir",
            str(build_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["cache_hit"] is True


def test_cli_compile_add_requires_llvm_path(monkeypatch, capsys):
    monkeypatch.delenv("MINITRITON_LLVM_BUILD_DIR", raising=False)

    rc = main(["compile-add"])

    assert rc == 2
    assert "--llvm-build-dir" in capsys.readouterr().out


def test_cli_cache_list_and_prune(tmp_path: Path, capsys):
    cache_root = (tmp_path / "artifacts" / "cache").resolve()
    cache_a = cache_root / "aaaa"
    cache_b = cache_root / "bbbb"
    cache_a.mkdir(parents=True)
    cache_b.mkdir(parents=True)

    (cache_a / "cache.json").write_text(
        json.dumps({"kernel": "add_kernel", "cuda_arch": "sm_80", "ptx_path": "a.ptx"}),
        encoding="utf-8",
    )
    (cache_b / "cache.json").write_text(
        json.dumps({"kernel": "add_kernel", "cuda_arch": "sm_80", "ptx_path": "b.ptx"}),
        encoding="utf-8",
    )

    list_rc = main(["cache-list", "--output-dir", str(tmp_path / "artifacts")])
    assert list_rc == 0
    list_payload = json.loads(capsys.readouterr().out)
    assert len(list_payload["entries"]) == 2

    prune_rc = main(["cache-prune", "--output-dir", str(tmp_path / "artifacts"), "--all"])
    assert prune_rc == 0
    prune_payload = json.loads(capsys.readouterr().out)
    assert len(prune_payload["removed"]) == 2

    list_rc_after = main(["cache-list", "--output-dir", str(tmp_path / "artifacts")])
    assert list_rc_after == 0
    list_payload_after = json.loads(capsys.readouterr().out)
    assert list_payload_after["entries"] == []


def test_cli_toolchain_check_reports_paths_and_nvptx(tmp_path: Path, capsys):
    build_dir = tmp_path / "llvm-build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True)

    _write_tool(bin_dir, "mlir-opt", "exit /b 0\n")
    _write_tool(bin_dir, "mlir-translate", "exit /b 0\n")
    _write_llc_with_targets(bin_dir, include_nvptx=True)

    rc = main(["toolchain-check", "--llvm-build-dir", str(build_dir)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["llc_nvptx_supported"] is True
    assert payload["tools"]["mlir_opt"]
    assert payload["tools"]["mlir_translate"]
    assert payload["tools"]["llc"]


def test_cli_toolchain_check_requires_llvm_path(monkeypatch, capsys):
    monkeypatch.delenv("MINITRITON_LLVM_BUILD_DIR", raising=False)

    rc = main(["toolchain-check"])

    assert rc == 2
    assert "--llvm-build-dir" in capsys.readouterr().out


def test_cli_launch_smoke_success(monkeypatch, tmp_path: Path, capsys):
    ptx_path = tmp_path / "noop.ptx"
    ptx_path.write_text(".visible .entry noop() { ret; }\n", encoding="utf-8")

    calls = {"load": 0, "launch": 0}

    def _fake_load_kernel(path, kernel_name):
        calls["load"] += 1
        assert str(path) == str(ptx_path)
        assert kernel_name == "noop"
        return object()

    def _fake_launch_kernel(kernel, *, arg_specs, args, launch_config):
        calls["launch"] += 1
        assert arg_specs == {}
        assert args == {}
        assert launch_config.grid_x == 1
        assert launch_config.block_x == 1

    monkeypatch.setattr("mini_triton.cli.load_kernel", _fake_load_kernel)
    monkeypatch.setattr("mini_triton.cli.launch_kernel", _fake_launch_kernel)

    rc = main(["launch-smoke", "--ptx-path", str(ptx_path), "--kernel-name", "noop"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["launched"] is True
    assert calls == {"load": 1, "launch": 1}


def test_cli_launch_smoke_missing_ptx(capsys):
    rc = main(["launch-smoke", "--ptx-path", "missing.ptx", "--kernel-name", "noop"])

    assert rc == 2
    assert "missing.ptx" in capsys.readouterr().out


def test_cli_vector_add_smoke_success(monkeypatch, tmp_path: Path, capsys):
    ptx_path = tmp_path / "add_kernel.ptx"
    ptx_path.write_text(".visible .entry add_kernel() { ret; }\n", encoding="utf-8")

    class _FakeCompileResult:
        def __init__(self, path: Path):
            self.cache_hit = False
            self.cache_key = "fake-key"
            self.ptx_path = path

    class _FakeKernel:
        def compile_cached(self, *_args, **_kwargs):
            return _FakeCompileResult(ptx_path)

    class _FakeDriver:
        def __init__(self):
            self._next = 0x1000
            self.memory: dict[int, bytes] = {}

        def mem_alloc(self, nbytes):
            from mini_triton.runtime import DevicePointer

            address = self._next
            self._next += int(nbytes)
            self.memory[address] = b"\x00" * int(nbytes)
            return DevicePointer(address)

        def mem_free(self, pointer):
            address = pointer.address if hasattr(pointer, "address") else int(pointer)
            self.memory.pop(address, None)

        def memcpy_htod(self, dst, src):
            self.memory[dst.address if hasattr(dst, "address") else int(dst)] = bytes(src)

        def memcpy_dtoh(self, src, _nbytes):
            return self.memory[src.address if hasattr(src, "address") else int(src)]

        def synchronize(self):
            return None

    class _FakeLoaded:
        def __init__(self):
            self.driver = _FakeDriver()

    fake_loaded = _FakeLoaded()

    def _fake_make_add_kernel(_block_size):
        return _FakeKernel()

    def _fake_load_kernel(_ptx, _kernel_name):
        return fake_loaded

    def _fake_launch_kernel(_kernel, *, arg_specs, args, launch_config):
        assert set(arg_specs.keys()) == {"x", "y", "out", "n"}
        assert launch_config.grid_x > 0
        assert launch_config.block_x > 0

        from array import array

        x_bytes = fake_loaded.driver.memory[args["x"].address]
        y_bytes = fake_loaded.driver.memory[args["y"].address]
        x_vals = array("f")
        y_vals = array("f")
        x_vals.frombytes(x_bytes)
        y_vals.frombytes(y_bytes)
        out_vals = array("f", [lhs + rhs for lhs, rhs in zip(x_vals, y_vals)])
        fake_loaded.driver.memory[args["out"].address] = out_vals.tobytes()

    monkeypatch.setattr("mini_triton.cli._make_add_kernel", _fake_make_add_kernel)
    monkeypatch.setattr("mini_triton.cli.load_kernel", _fake_load_kernel)
    monkeypatch.setattr("mini_triton.cli.launch_kernel", _fake_launch_kernel)
    monkeypatch.setattr("mini_triton.cli.CtypesCudaDriver", _FakeDriver)

    rc = main(
        [
            "vector-add-smoke",
            "--llvm-build-dir",
            str(tmp_path / "llvm-build"),
            "--num-elements",
            "16",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["num_elements"] == 16
    assert payload["max_abs_error"] == 0.0


def test_cli_vector_add_smoke_validates_num_elements(capsys):
    rc = main(["vector-add-smoke", "--llvm-build-dir", "x", "--num-elements", "0"])

    assert rc == 2
    assert "--num-elements" in capsys.readouterr().out


def test_cli_vector_add_smoke_rejects_mismatched_ranges(capsys):
    rc = main(
        [
            "vector-add-smoke",
            "--llvm-build-dir",
            "x",
            "--x-range",
            "1:1000",
            "--y-range",
            "1000:2000",
        ]
    )

    assert rc == 2
    assert "same number of elements" in capsys.readouterr().out


def _write_tool(bin_dir: Path, tool_name: str, body: str) -> None:
    if os.name == "nt":
        path = bin_dir / f"{tool_name}.cmd"
        path.write_text("@echo off\n" + body, encoding="utf-8")
        return

    path = bin_dir / tool_name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR)


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


def _write_llc_with_targets(bin_dir: Path, *, include_nvptx: bool) -> None:
    if os.name == "nt":
        nvptx_line = "  echo   nvptx - NVPTX 32-bit\n" if include_nvptx else ""
        _write_tool(
            bin_dir,
            "llc",
            "if \"%~1\"==\"--version\" (\n"
            "  echo LLVM fake\n"
            "  echo Registered Targets:\n"
            "  echo   x86 - 64-bit X86\n"
            f"{nvptx_line}"
            "  exit /b 0\n"
            ")\n"
            "exit /b 0\n",
        )
        return

    nvptx_line = "echo '  nvptx - NVPTX 32-bit'\n" if include_nvptx else ""
    _write_tool(
        bin_dir,
        "llc",
        "if [ \"${1:-}\" = \"--version\" ]; then\n"
        "  echo 'LLVM fake'\n"
        "  echo 'Registered Targets:'\n"
        "  echo '  x86 - 64-bit X86'\n"
        f"  {nvptx_line}"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
    )
