from __future__ import annotations

from dataclasses import dataclass

from .ir import ArgSpec, KernelArg, KernelIR, StoreNode, ValueNode, stringify_attr


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolicValue:
    session: "CaptureSession"
    name: str
    dtype: str

    def __add__(self, other: object) -> "SymbolicValue":
        other_value = self.session.expect_symbolic_value(other, op_name="add")
        return self.session.add_value("add", self.dtype, [self, other_value])

    def __lt__(self, other: object) -> "SymbolicValue":
        other_value = self.session.expect_symbolic_value(other, op_name="lt")
        return self.session.add_value("cmp_lt", "pred", [self, other_value])

    def __bool__(self) -> bool:
        raise CaptureError(
            "symbolic values cannot be converted to bool during capture; "
            "use explicit mask-producing operations instead"
        )


class KernelContextProxy:
    def __init__(self, session: "CaptureSession") -> None:
        self._session = session

    def program_id(self, axis: int = 0) -> SymbolicValue:
        if axis != 0:
            raise CaptureError(f"program_id currently supports axis=0 only, got axis={axis}")
        return self._session.add_value(
            "program_id",
            "index",
            [],
            axis=axis,
            scope="global",
        )

    def global_index(self) -> SymbolicValue:
        return self.program_id(axis=0)

    def arange(self, start: int, end: int) -> SymbolicValue:
        if not isinstance(start, int) or not isinstance(end, int):
            raise CaptureError("arange expects integer start/end values")
        if end <= start:
            raise CaptureError(f"arange expects end > start, got start={start}, end={end}")

        return self._session.add_value(
            "arange",
            "index",
            [],
            start=start,
            end=end,
        )


class BufferProxy:
    def __init__(self, session: "CaptureSession", arg_name: str, spec: ArgSpec) -> None:
        self._session = session
        self._arg_name = arg_name
        self._spec = spec

    def load(
        self,
        index: SymbolicValue,
        mask: SymbolicValue | None = None,
    ) -> SymbolicValue:
        index_value = self._session.expect_symbolic_value(index, expected_dtype="index", op_name="load")
        mask_value = self._session.optional_predicate(mask, op_name="load")
        inputs = [self._session.arg_value(self._arg_name), index_value]
        if mask_value is not None:
            inputs.append(mask_value)
        return self._session.add_value("load", self._spec.dtype, inputs, source=self._arg_name)

    def store(
        self,
        index: SymbolicValue,
        value: SymbolicValue,
        mask: SymbolicValue | None = None,
    ) -> None:
        index_value = self._session.expect_symbolic_value(index, expected_dtype="index", op_name="store")
        stored_value = self._session.expect_symbolic_value(value, expected_dtype=self._spec.dtype, op_name="store")
        mask_value = self._session.optional_predicate(mask, op_name="store")
        self._session.add_store(self._arg_name, index_value, stored_value, mask_value)


class ScalarProxy(SymbolicValue):
    pass


class CaptureSession:
    def __init__(self, kernel_name: str, block_size: int, arg_specs: dict[str, ArgSpec]) -> None:
        self._counter = 0
        self._arg_values: dict[str, SymbolicValue] = {}
        self.ir = KernelIR(name=kernel_name, block_size=block_size)

        for arg_name, spec in arg_specs.items():
            self.ir.args.append(KernelArg(name=arg_name, spec=spec))
            self._arg_values[arg_name] = SymbolicValue(self, arg_name, spec.dtype)

    def arg_value(self, arg_name: str) -> SymbolicValue:
        return self._arg_values[arg_name]

    def add_value(
        self,
        op: str,
        dtype: str,
        inputs: list[SymbolicValue],
        **attrs: object,
    ) -> SymbolicValue:
        value_name = self._next_value_name()
        rendered_attrs = tuple(
            sorted((key, stringify_attr(value)) for key, value in attrs.items())
        )
        node = ValueNode(
            name=value_name,
            op=op,
            dtype=dtype,
            inputs=tuple(input_value.name for input_value in inputs),
            attrs=rendered_attrs,
        )
        self.ir.values.append(node)
        return SymbolicValue(self, value_name, dtype)

    def add_store(
        self,
        buffer_name: str,
        index: SymbolicValue,
        value: SymbolicValue,
        mask: SymbolicValue | None,
    ) -> None:
        self.ir.stores.append(
            StoreNode(
                buffer=buffer_name,
                index=index.name,
                value=value.name,
                mask=mask.name if mask is not None else None,
            )
        )

    def expect_symbolic_value(
        self,
        value: object,
        expected_dtype: str | None = None,
        op_name: str = "operation",
    ) -> SymbolicValue:
        if not isinstance(value, SymbolicValue):
            raise CaptureError(f"{op_name} expects symbolic values, got {type(value).__name__}")

        if expected_dtype is not None and value.dtype != expected_dtype:
            raise CaptureError(
                f"{op_name} expects dtype {expected_dtype}, got {value.dtype}"
            )

        return value

    def optional_predicate(
        self,
        value: SymbolicValue | None,
        op_name: str,
    ) -> SymbolicValue | None:
        if value is None:
            return None
        return self.expect_symbolic_value(value, expected_dtype="pred", op_name=op_name)

    def _next_value_name(self) -> str:
        name = f"v{self._counter}"
        self._counter += 1
        return name