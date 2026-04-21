from __future__ import annotations

from .ir import KernelIR


def emit_mini_dialect(kernel_ir: KernelIR) -> str:
    lines: list[str] = [f"kernel {kernel_ir.name} {kernel_ir.block_size}"]

    for arg in kernel_ir.args:
        lines.append(
            " ".join(
                [
                    "arg",
                    arg.name,
                    arg.spec.kind,
                    arg.spec.dtype,
                    arg.spec.address_space,
                ]
            )
        )

    for value in kernel_ir.values:
        inputs = ",".join(value.inputs) if value.inputs else "-"
        attrs = ",".join(f"{key}={raw_value}" for key, raw_value in value.attrs) if value.attrs else "-"
        lines.append(" ".join(["value", value.name, value.op, value.dtype, inputs, attrs]))

    for store in kernel_ir.stores:
        mask = store.mask if store.mask is not None else "-"
        lines.append(" ".join(["store", store.buffer, store.index, store.value, mask]))

    return "\n".join(lines)
