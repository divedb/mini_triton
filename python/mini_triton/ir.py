from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArgSpec:
    kind: str
    dtype: str
    address_space: str = "global"

    def describe(self) -> str:
        if self.kind == "buffer":
            return f"buffer<{self.dtype}, {self.address_space}>"
        return f"scalar<{self.dtype}>"


@dataclass(frozen=True)
class KernelArg:
    name: str
    spec: ArgSpec


@dataclass(frozen=True)
class ValueNode:
    name: str
    op: str
    dtype: str
    inputs: tuple[str, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StoreNode:
    buffer: str
    index: str
    value: str
    mask: str | None = None


@dataclass
class KernelIR:
    name: str
    block_size: int
    args: list[KernelArg] = field(default_factory=list)
    values: list[ValueNode] = field(default_factory=list)
    stores: list[StoreNode] = field(default_factory=list)

    def format(self) -> str:
        lines = [f"kernel @{self.name} [block_size={self.block_size}]"]

        for arg in self.args:
            lines.append(f"  arg %{arg.name}: {arg.spec.describe()}")

        for value in self.values:
            rendered_inputs = ", ".join(f"%{input_name}" for input_name in value.inputs)
            rendered_attrs = "; ".join(f"{key}={raw_value}" for key, raw_value in value.attrs)

            pieces = [piece for piece in [rendered_inputs, rendered_attrs] if piece]
            suffix = f" ({'; '.join(pieces)})" if pieces else ""
            lines.append(f"  %{value.name} = {value.op} : {value.dtype}{suffix}")

        for store in self.stores:
            mask_suffix = f", mask=%{store.mask}" if store.mask else ""
            lines.append(
                f"  store %{store.buffer}[%{store.index}] <- %{store.value}{mask_suffix}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def stringify_attr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)