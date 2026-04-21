from mini_triton import buffer, kernel, scalar


@kernel(block_size=128)
def add_kernel(ctx, x, y, out, n):
    idx = ctx.global_index()
    active = idx < n
    out.store(idx, x.load(idx, active) + y.load(idx, active), active)


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