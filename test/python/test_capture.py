from mini_triton import buffer, kernel, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


@kernel(block_size=128)
def mul_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) * y.load(idx, active), active)


def test_vector_add_capture_is_deterministic():
    captured = add_kernel.capture(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )

    assert (
        captured.format()
        == "\n".join(
            [
                "kernel @add_kernel [block_size=128]",
                "  arg %x: buffer<float32, global>",
                "  arg %y: buffer<float32, global>",
                "  arg %out: buffer<float32, global>",
                "  arg %n: scalar<index>",
                "  %v0 = program_id : index (axis=0; scope='global')",
                "  %v1 = cmp_lt : pred (%v0, %n)",
                "  %v2 = load : float32 (%x, %v0, %v1; source='x')",
                "  %v3 = load : float32 (%y, %v0, %v1; source='y')",
                "  %v4 = add : float32 (%v2, %v3)",
                "  store %out[%v0] <- %v4, mask=%v1",
            ]
        )
    )


def test_vector_mul_capture_records_mul_node():
    captured = mul_kernel.capture(
        x=buffer("float32"),
        y=buffer("float32"),
        out=buffer("float32"),
        n=scalar("index"),
    )
    assert " = mul : float32 (" in captured.format()

    assert "store %out[%v0] <- %v4, mask=%v1" in captured.format()


def test_capture_requires_declared_argument_order():
    try:
        add_kernel.capture(
            y=buffer("float32"),
            x=buffer("float32"),
            out=buffer("float32"),
            n=scalar("index"),
        )
    except Exception as exc:  # noqa: BLE001
        assert "declared parameter order" in str(exc)
    else:
        raise AssertionError("expected capture to reject argument spec reordering")


@kernel(block_size=64)
def program_id_kernel(ctx, out, n):
    pid = ctx.program_id(axis=0)
    active = pid < n
    out.store(pid, pid, active)


def test_program_id_capture_uses_axis_zero():
    captured = program_id_kernel.capture(
        out=buffer("index"),
        n=scalar("index"),
    )
    assert "program_id : index (axis=0; scope='global')" in captured.format()


@kernel(block_size=32)
def arange_kernel(ctx, out):
    idx = ctx.arange(0, 16)
    out.store(idx, idx)


@kernel(block_size=32)
def arange_step_kernel(ctx, out):
    idx = ctx.arange(0, 16, 2)
    out.store(idx, idx)


def test_arange_capture_records_bounds():
    captured = arange_kernel.capture(
        out=buffer("index"),
    )
    rendered = captured.format()
    assert "arange : index" in rendered
    assert "start=0" in rendered
    assert "end=16" in rendered
    assert "step=1" in rendered


def test_arange_capture_records_step():
    captured = arange_step_kernel.capture(
        out=buffer("index"),
    )
    rendered = captured.format()
    assert "arange : index" in rendered
    assert "step=2" in rendered


def test_program_id_rejects_out_of_range_axis_legacy_case():
    @kernel(block_size=32)
    def invalid_axis_kernel(ctx, out):
        idx = ctx.program_id(axis=3)
        out.store(idx, idx)

    try:
        invalid_axis_kernel.capture(out=buffer("index"))
    except Exception as exc:  # noqa: BLE001
        assert "axis=0 or axis=1 only" in str(exc)
    else:
        raise AssertionError("expected out-of-range program_id axis to fail")


def test_arange_rejects_invalid_bounds():
    @kernel(block_size=32)
    def invalid_arange_kernel(ctx, out):
        idx = ctx.arange(16, 16)
        out.store(idx, idx)

    try:
        invalid_arange_kernel.capture(out=buffer("index"))
    except Exception as exc:  # noqa: BLE001
        assert "end > start" in str(exc)
    else:
        raise AssertionError("expected invalid arange bounds to fail")


def test_arange_rejects_invalid_step():
    @kernel(block_size=32)
    def invalid_arange_step_kernel(ctx, out):
        idx = ctx.arange(0, 16, 0)
        out.store(idx, idx)

    try:
        invalid_arange_step_kernel.capture(out=buffer("index"))
    except Exception as exc:  # noqa: BLE001
        assert "step > 0" in str(exc)
    else:
        raise AssertionError("expected invalid arange step to fail")


@kernel(block_size=32)
def block_scope_kernel(ctx, out):
    idx = ctx.arange(0, 32, 1, scope="block")
    out.store(idx, idx)


def test_arange_block_scope_is_captured():
    captured = block_scope_kernel.capture(out=buffer("index"))
    rendered = captured.format()
    assert "scope='block'" in rendered


@kernel(block_size=32)
def axis1_kernel(ctx, out):
    idx = ctx.program_id(axis=1)
    out.store(idx, idx)


def test_program_id_axis1_is_captured():
    captured = axis1_kernel.capture(out=buffer("index"))
    assert "program_id : index (axis=1; scope='global')" in captured.format()


def test_program_id_rejects_axis_out_of_range():
    @kernel(block_size=32)
    def invalid_axis2_kernel(ctx, out):
        idx = ctx.program_id(axis=2)
        out.store(idx, idx)

    try:
        invalid_axis2_kernel.capture(out=buffer("index"))
    except Exception as exc:  # noqa: BLE001
        assert "axis=0 or axis=1 only" in str(exc)
    else:
        raise AssertionError("expected out-of-range program_id axis to fail")