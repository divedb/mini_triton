from mini_triton import buffer, emit_mlir, kernel, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


def test_vector_add_mlir_is_deterministic():
    captured = add_kernel.capture(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert emit_mlir(captured) == "\n".join(
        [
            "module {",
            "  llvm.func @add_kernel(%x: !llvm.ptr, %y: !llvm.ptr, %out: !llvm.ptr, %n: i64) attributes {nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>, mini_triton.block_size = 128 : i32} {",
            "    %v0_block_id = nvvm.read.ptx.sreg.ctaid.x : i32",
            "    %v0_block_dim = nvvm.read.ptx.sreg.ntid.x : i32",
            "    %v0_thread_id = nvvm.read.ptx.sreg.tid.x : i32",
            "    %v0_block_id_i64 = llvm.sext %v0_block_id : i32 to i64",
            "    %v0_block_dim_i64 = llvm.sext %v0_block_dim : i32 to i64",
            "    %v0_thread_id_i64 = llvm.sext %v0_thread_id : i32 to i64",
            "    %v0_block_base = llvm.mul %v0_block_id_i64, %v0_block_dim_i64 : i64",
            "    %v0 = llvm.add %v0_block_base, %v0_thread_id_i64 : i64",
            "    %v1 = llvm.icmp \"slt\" %v0, %n : i64",
            "    llvm.cond_br %v1, ^bb1, ^bb2",
            "  ^bb1:",
            "    %v2_ptr = llvm.getelementptr %x[%v0] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "    %v2 = llvm.load %v2_ptr : !llvm.ptr -> f32",
            "    %v3_ptr = llvm.getelementptr %y[%v0] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "    %v3 = llvm.load %v3_ptr : !llvm.ptr -> f32",
            "    %v4 = llvm.fadd %v2, %v3 : f32",
            "    %v4_store_ptr = llvm.getelementptr %out[%v0] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
            "    llvm.store %v4, %v4_store_ptr : f32, !llvm.ptr",
            "    llvm.br ^bb2",
            "  ^bb2:",
            "    llvm.return",
            "  }",
            "}",
        ]
    )


def test_kernel_can_emit_mlir_directly():
    assert add_kernel.emit_mlir(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    ).startswith("module {")