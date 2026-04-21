from __future__ import annotations

import os
from array import array
from pathlib import Path

from mini_triton import (
    CtypesCudaDriver,
    LaunchConfig,
    buffer,
    kernel,
    launch_kernel,
    load_kernel,
    scalar,
)


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


def _resolve_llvm_build_dir() -> str:
    llvm_build_dir = os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if llvm_build_dir:
        return llvm_build_dir

    repo_root = Path(__file__).resolve().parents[1]
    default_llvm_build = repo_root / "third_party" / "llvm-project" / "build"
    if default_llvm_build.is_dir():
        return str(default_llvm_build)

    raise RuntimeError(
        "set MINITRITON_LLVM_BUILD_DIR or create third_party/llvm-project/build before running this example"
    )


def main() -> None:
    # User-provided example inputs.
    x_values = list(range(1, 1001))
    y_values = list(range(1000, 2000))

    if len(x_values) != len(y_values):
        raise ValueError(f"x and y must have the same length, got {len(x_values)} and {len(y_values)}")

    llvm_build_dir = _resolve_llvm_build_dir()

    compile_result = add_kernel.compile_cached(
        Path("build") / "mvp_artifacts_real",
        llvm_build_dir=llvm_build_dir,
        cuda_arch="sm_80",
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    loaded = load_kernel(compile_result.ptx_path, "add_kernel")
    driver = loaded.driver
    if not isinstance(driver, CtypesCudaDriver):
        raise RuntimeError("runtime driver does not support device memory operations")

    host_x = array("f", [float(value) for value in x_values])
    host_y = array("f", [float(value) for value in y_values])
    expected = array("f", [lhs + rhs for lhs, rhs in zip(host_x, host_y)])
    n = len(host_x)
    nbytes = n * host_x.itemsize

    device_x = driver.mem_alloc(nbytes)
    device_y = driver.mem_alloc(nbytes)
    device_out = driver.mem_alloc(nbytes)

    try:
        driver.memcpy_htod(device_x, host_x.tobytes())
        driver.memcpy_htod(device_y, host_y.tobytes())

        launch_kernel(
            loaded,
            arg_specs={
                "x": buffer("float32"),
                "y": buffer("float32"),
                "out": buffer("float32"),
                "n": scalar("index"),
            },
            args={"x": device_x, "y": device_y, "out": device_out, "n": n},
            launch_config=LaunchConfig.for_num_elements(n, add_kernel.block_size),
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
    if max_abs_error > 1e-5:
        raise RuntimeError(f"vector add verification failed with max_abs_error={max_abs_error}")

    result = list(host_out)
    print(f"n={n}")
    print(f"cache_hit={compile_result.cache_hit}")
    print(f"ptx_path={compile_result.ptx_path}")
    print(f"max_abs_error={max_abs_error}")
    print(f"result_head={result[:5]}")
    print(f"result_tail={result[-5:]}")


if __name__ == "__main__":
    main()
