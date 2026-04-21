from __future__ import annotations

import argparse
from array import array
import json
import os
import shutil
import subprocess
from pathlib import Path

from .api import buffer, kernel, scalar
from .compiler import CompilerConfigError, inspect_llc_nvptx_support, resolve_toolchain
from .runtime import CtypesCudaDriver, CudaDriverError, LaunchConfig, RuntimeUnavailableError, load_kernel, launch_kernel


def _make_add_kernel(block_size: int):
    @kernel(block_size=block_size)
    def add_kernel(ctx, x, y, out, n):
        idx = ctx.global_index()
        active = idx < n
        out.store(idx, x.load(idx, active) + y.load(idx, active), active)

    return add_kernel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini-triton")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_add = subparsers.add_parser("compile-add", help="Compile the MVP vector-add kernel")
    compile_add.add_argument("--output-dir", default="build/mvp_artifacts")
    compile_add.add_argument("--llvm-build-dir", default=None)
    compile_add.add_argument("--cpp-lowering-driver", default=None)
    compile_add.add_argument("--cuda-arch", default="sm_80")
    compile_add.add_argument("--block-size", type=int, default=128)
    compile_add.add_argument("--no-cache", action="store_true")

    cache_list = subparsers.add_parser("cache-list", help="List compile cache entries")
    cache_list.add_argument("--output-dir", default="build/mvp_artifacts")

    cache_prune = subparsers.add_parser("cache-prune", help="Prune compile cache entries")
    cache_prune.add_argument("--output-dir", default="build/mvp_artifacts")
    cache_prune.add_argument("--keep", type=int, default=5)
    cache_prune.add_argument("--all", action="store_true")

    toolchain_check = subparsers.add_parser("toolchain-check", help="Inspect configured LLVM/MLIR toolchain")
    toolchain_check.add_argument("--llvm-build-dir", default=None)
    toolchain_check.add_argument("--cpp-lowering-driver", default=None)

    launch_smoke = subparsers.add_parser("launch-smoke", help="Load and launch a PTX kernel for runtime smoke testing")
    launch_smoke.add_argument("--ptx-path", required=True)
    launch_smoke.add_argument("--kernel-name", required=True)
    launch_smoke.add_argument("--grid-x", type=int, default=1)
    launch_smoke.add_argument("--block-x", type=int, default=1)

    vector_add_smoke = subparsers.add_parser("vector-add-smoke", help="Compile and execute vector_add on GPU")
    vector_add_smoke.add_argument("--llvm-build-dir", default=None)
    vector_add_smoke.add_argument("--cpp-lowering-driver", default=None)
    vector_add_smoke.add_argument("--output-dir", default="build/mvp_artifacts_real")
    vector_add_smoke.add_argument("--cuda-arch", default="sm_80")
    vector_add_smoke.add_argument("--block-size", type=int, default=128)
    vector_add_smoke.add_argument("--num-elements", type=int, default=1024)
    vector_add_smoke.add_argument(
        "--x-range",
        default=None,
        help="Optional Python-style range start:end for x values (end-exclusive)",
    )
    vector_add_smoke.add_argument(
        "--y-range",
        default=None,
        help="Optional Python-style range start:end for y values (end-exclusive)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compile-add":
        return _run_compile_add(args)
    if args.command == "cache-list":
        return _run_cache_list(args)
    if args.command == "cache-prune":
        return _run_cache_prune(args)
    if args.command == "toolchain-check":
        return _run_toolchain_check(args)
    if args.command == "launch-smoke":
        return _run_launch_smoke(args)
    if args.command == "vector-add-smoke":
        return _run_vector_add_smoke(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_compile_add(args: argparse.Namespace) -> int:
    llvm_build_dir = args.llvm_build_dir or os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if not llvm_build_dir:
        print("error: set --llvm-build-dir or MINITRITON_LLVM_BUILD_DIR", flush=True)
        return 2

    add_kernel = _make_add_kernel(args.block_size)
    arg_specs = {
        "x": buffer("float32"),
        "y": buffer("float32"),
        "out": buffer("float32"),
        "n": scalar("index"),
    }

    output_dir = Path(args.output_dir)
    try:
        if args.no_cache:
            run = add_kernel.compile(
                output_dir,
                llvm_build_dir=llvm_build_dir,
                cpp_lowering_driver=args.cpp_lowering_driver,
                cuda_arch=args.cuda_arch,
                **arg_specs,
            )
            payload = {
                "kernel": "add_kernel",
                "cache_hit": False,
                "ptx_path": str(run.plan.artifacts.ptx),
                "output_dir": str(output_dir),
                "cuda_arch": args.cuda_arch,
                "cpp_lowering_driver": args.cpp_lowering_driver,
            }
        else:
            result = add_kernel.compile_cached(
                output_dir,
                llvm_build_dir=llvm_build_dir,
                cpp_lowering_driver=args.cpp_lowering_driver,
                cuda_arch=args.cuda_arch,
                **arg_specs,
            )
            payload = {
                "kernel": "add_kernel",
                "cache_hit": result.cache_hit,
                "cache_key": result.cache_key,
                "ptx_path": str(result.ptx_path),
                "output_dir": str(output_dir),
                "cuda_arch": args.cuda_arch,
                "cpp_lowering_driver": args.cpp_lowering_driver,
            }
    except CompilerConfigError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            print(stderr, flush=True)
        else:
            print(f"error: command failed with exit code {exc.returncode}", flush=True)
        return 1

    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_cache_list(args: argparse.Namespace) -> int:
    cache_root = _cache_root_from_output_dir(Path(args.output_dir))
    entries = _collect_cache_entries(cache_root)
    payload = {
        "cache_root": str(cache_root),
        "entries": entries,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_cache_prune(args: argparse.Namespace) -> int:
    cache_root = _cache_root_from_output_dir(Path(args.output_dir))
    cache_dirs = _cache_dirs(cache_root)

    if args.all:
        to_remove = cache_dirs
        kept = []
    else:
        keep = max(int(args.keep), 0)
        cache_dirs = sorted(cache_dirs, key=lambda path: path.stat().st_mtime_ns, reverse=True)
        kept = cache_dirs[:keep]
        to_remove = cache_dirs[keep:]

    for cache_dir in to_remove:
        shutil.rmtree(cache_dir)

    payload = {
        "cache_root": str(cache_root),
        "removed": [str(path) for path in to_remove],
        "kept": [str(path) for path in kept],
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_toolchain_check(args: argparse.Namespace) -> int:
    llvm_build_dir = args.llvm_build_dir or os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if not llvm_build_dir:
        print("error: set --llvm-build-dir or MINITRITON_LLVM_BUILD_DIR", flush=True)
        return 2

    try:
        toolchain = resolve_toolchain(llvm_build_dir, cpp_lowering_driver=args.cpp_lowering_driver)
    except CompilerConfigError as exc:
        print(f"error: {exc}", flush=True)
        return 2

    llc_nvptx_supported, llc_targets = inspect_llc_nvptx_support(toolchain.llc)
    payload = {
        "llvm_build_dir": str(toolchain.llvm_build_dir),
        "tools": {
            "mlir_opt": str(toolchain.mlir_opt),
            "mlir_translate": str(toolchain.mlir_translate),
            "llc": str(toolchain.llc),
        },
        "cpp_lowering_driver": str(toolchain.cpp_lowering_driver) if toolchain.cpp_lowering_driver else None,
        "llc_nvptx_supported": llc_nvptx_supported,
        "llc_registered_targets": llc_targets,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_launch_smoke(args: argparse.Namespace) -> int:
    try:
        kernel = load_kernel(args.ptx_path, args.kernel_name)
        launch_kernel(
            kernel,
            arg_specs={},
            args={},
            launch_config=LaunchConfig(grid_x=args.grid_x, block_x=args.block_x),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", flush=True)
        return 2
    except (RuntimeUnavailableError, CudaDriverError) as exc:
        print(f"error: {exc}", flush=True)
        return 3

    payload = {
        "ptx_path": str(Path(args.ptx_path).expanduser().resolve()),
        "kernel_name": args.kernel_name,
        "grid_x": args.grid_x,
        "block_x": args.block_x,
        "launched": True,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _run_vector_add_smoke(args: argparse.Namespace) -> int:
    llvm_build_dir = args.llvm_build_dir or os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if not llvm_build_dir:
        print("error: set --llvm-build-dir or MINITRITON_LLVM_BUILD_DIR", flush=True)
        return 2
    if args.num_elements <= 0:
        print("error: --num-elements must be positive", flush=True)
        return 2

    if (args.x_range is None) != (args.y_range is None):
        print("error: provide both --x-range and --y-range together", flush=True)
        return 2

    add_kernel = _make_add_kernel(args.block_size)
    arg_specs = {
        "x": buffer("float32"),
        "y": buffer("float32"),
        "out": buffer("float32"),
        "n": scalar("index"),
    }

    if args.x_range is not None and args.y_range is not None:
        try:
            x_values = _parse_int_range(args.x_range, "x-range")
            y_values = _parse_int_range(args.y_range, "y-range")
        except ValueError as exc:
            print(f"error: {exc}", flush=True)
            return 2
        if len(x_values) != len(y_values):
            print(
                "error: --x-range and --y-range must produce the same number of elements "
                f"(got {len(x_values)} and {len(y_values)})",
                flush=True,
            )
            return 2
        if not x_values:
            print("error: provided ranges produced zero elements", flush=True)
            return 2

        host_x = array("f", [float(value) for value in x_values])
        host_y = array("f", [float(value) for value in y_values])
        n = len(host_x)
    else:
        n = int(args.num_elements)
        host_x = array("f", [float(index) for index in range(n)])
        host_y = array("f", [float(index * 2) for index in range(n)])

    expected = array("f", [x + y for x, y in zip(host_x, host_y)])
    nbytes = len(host_x) * host_x.itemsize

    try:
        compile_result = add_kernel.compile_cached(
            Path(args.output_dir),
            llvm_build_dir=llvm_build_dir,
            cpp_lowering_driver=args.cpp_lowering_driver,
            cuda_arch=args.cuda_arch,
            **arg_specs,
        )
        loaded = load_kernel(compile_result.ptx_path, "add_kernel")
    except (CompilerConfigError, RuntimeUnavailableError, CudaDriverError) as exc:
        print(f"error: {exc}", flush=True)
        return 3
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(stderr if stderr else f"error: command failed with exit code {exc.returncode}", flush=True)
        return 1

    driver = loaded.driver
    if not isinstance(driver, CtypesCudaDriver):
        print("error: loaded runtime driver does not support device memory operations", flush=True)
        return 3

    device_x = driver.mem_alloc(nbytes)
    device_y = driver.mem_alloc(nbytes)
    device_out = driver.mem_alloc(nbytes)

    try:
        driver.memcpy_htod(device_x, host_x.tobytes())
        driver.memcpy_htod(device_y, host_y.tobytes())

        launch_kernel(
            loaded,
            arg_specs=arg_specs,
            args={
                "x": device_x,
                "y": device_y,
                "out": device_out,
                "n": n,
            },
            launch_config=LaunchConfig.for_num_elements(n, args.block_size),
        )
        driver.synchronize()

        out_bytes = driver.memcpy_dtoh(device_out, nbytes)
        host_out = array("f")
        host_out.frombytes(out_bytes)
    finally:
        driver.mem_free(device_out)
        driver.mem_free(device_y)
        driver.mem_free(device_x)

    max_abs_error = max(abs(a - b) for a, b in zip(host_out, expected)) if n else 0.0
    verified = max_abs_error <= 1e-5
    if not verified:
        print(json.dumps({"verified": False, "max_abs_error": max_abs_error}, indent=2), flush=True)
        return 4

    payload = {
        "verified": True,
        "num_elements": n,
        "max_abs_error": max_abs_error,
        "cache_hit": compile_result.cache_hit,
        "cache_key": compile_result.cache_key,
        "ptx_path": str(compile_result.ptx_path),
        "sample_out": [host_out[0], host_out[min(1, n - 1)], host_out[min(2, n - 1)]],
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def _cache_root_from_output_dir(output_dir: Path) -> Path:
    return output_dir.expanduser().resolve() / "cache"


def _cache_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    return [path for path in cache_root.iterdir() if path.is_dir()]


def _collect_cache_entries(cache_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for cache_dir in sorted(_cache_dirs(cache_root), key=lambda path: path.name):
        entry = {
            "cache_key": cache_dir.name,
            "cache_dir": str(cache_dir),
            "metadata_present": False,
        }

        metadata_path = cache_dir / "cache.json"
        if metadata_path.is_file():
            entry["metadata_present"] = True
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(metadata, dict):
                    entry.update(metadata)
            except json.JSONDecodeError:
                entry["metadata_error"] = "invalid_json"

        entries.append(entry)

    return entries


def _parse_int_range(raw: str, arg_name: str) -> range:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"{arg_name} must be in start:end format")

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{arg_name} values must be integers") from exc

    if end < start:
        raise ValueError(f"{arg_name} end must be >= start")

    return range(start, end)


if __name__ == "__main__":
    raise SystemExit(main())
