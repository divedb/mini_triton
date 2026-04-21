from __future__ import annotations

import os
from pathlib import Path

from mini_triton import buffer, kernel, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


def main() -> None:
    llvm_build_dir = os.environ.get("MINITRITON_LLVM_BUILD_DIR")
    if not llvm_build_dir:
        repo_root = Path(__file__).resolve().parents[1]
        default_llvm_build = repo_root / "third_party" / "llvm-project" / "build"
        if default_llvm_build.is_dir():
            llvm_build_dir = str(default_llvm_build)
        else:
            raise RuntimeError(
                "set MINITRITON_LLVM_BUILD_DIR or create third_party/llvm-project/build before running this example"
            )

    result = add_kernel.compile_cached(
        Path("build") / "mvp_artifacts",
        llvm_build_dir=llvm_build_dir,
        cuda_arch="sm_80",
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    print(f"cache_key={result.cache_key}")
    print(f"cache_hit={result.cache_hit}")
    print(f"ptx_path={result.ptx_path}")


if __name__ == "__main__":
    main()
