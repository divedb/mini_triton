from mini_triton import buffer, emit_mlir, kernel, scalar


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


@kernel(block_size=64)
def add_kernel_with_block_scope_arange(ctx, x, y, out, n):
    idx = ctx.arange(0, 1024, 1, scope="block")
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


@kernel(block_size=64)
def axis1_program_id_kernel(ctx, out):
    idx = ctx.program_id(axis=1)
    out.store(idx, idx)


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


def test_arange_mlir_contains_start_offset_logic():
    mlir_text = add_kernel_with_arange.emit_mlir(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert "llvm.mlir.constant(1 : i64) : i64" in mlir_text
    assert "= llvm.add %v0_base, %v0_start : i64" in mlir_text


def test_arange_step_mlir_contains_step_multiply_logic():
    mlir_text = add_kernel_with_arange_step.emit_mlir(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert "llvm.mlir.constant(2 : i64) : i64" in mlir_text
    assert "= llvm.mul %v0_base, %v0_step : i64" in mlir_text
    assert "= llvm.add %v0_stepped, %v0_start : i64" in mlir_text


def test_block_scope_arange_mlir_uses_tid_scope():
    mlir_text = add_kernel_with_block_scope_arange.emit_mlir(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert "nvvm.read.ptx.sreg.tid.x" in mlir_text
    assert "nvvm.read.ptx.sreg.ctaid.x" not in mlir_text


def test_axis1_program_id_mlir_uses_y_registers():
    mlir_text = axis1_program_id_kernel.emit_mlir(out=buffer("index"))
    assert "nvvm.read.ptx.sreg.ctaid.y" in mlir_text
    assert "nvvm.read.ptx.sreg.ntid.y" in mlir_text
    assert "nvvm.read.ptx.sreg.tid.y" in mlir_text