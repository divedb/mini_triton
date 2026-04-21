from __future__ import annotations

from dataclasses import dataclass

from .ir import ArgSpec, KernelArg, KernelIR, StoreNode, ValueNode


class MLIREmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmittedKernel:
    ir: KernelIR
    module_text: str


def emit_mlir(kernel_ir: KernelIR) -> str:
    return MLIRBuilder(kernel_ir).build_module()


class MLIRBuilder:
    def __init__(self, kernel_ir: KernelIR) -> None:
        self.kernel_ir = kernel_ir
        self._value_nodes = {value.name: value for value in kernel_ir.values}
        self._arg_specs = {arg.name: arg.spec for arg in kernel_ir.args}

    def build_module(self) -> str:
        guard_name = self._determine_common_mask()
        guarded_values = self._determine_guarded_values(guard_name)

        lines = ["module {"]
        lines.append(
            "  llvm.func @"
            f"{self.kernel_ir.name}({self._format_signature()}) attributes "
            f"{{nvvm.kernel, nvvm.maxntid = array<i32: {self.kernel_ir.block_size}, 1, 1>, "
            f"mini_triton.block_size = {self.kernel_ir.block_size} : i32}} {{"
        )

        for value in self.kernel_ir.values:
            if value.name in guarded_values:
                continue
            lines.extend(self._indent_lines(self._emit_value(value), "    "))

        if guard_name is not None:
            lines.append(f"    llvm.cond_br %{guard_name}, ^bb1, ^bb2")
            lines.append("  ^bb1:")
            for value in self.kernel_ir.values:
                if value.name in guarded_values:
                    lines.extend(self._indent_lines(self._emit_value(value), "    "))
            for store in self.kernel_ir.stores:
                if store.mask is None:
                    raise MLIREmissionError("mixed masked and unmasked stores are not supported")
                lines.extend(self._indent_lines(self._emit_store(store), "    "))
            lines.append("    llvm.br ^bb2")
            lines.append("  ^bb2:")
        else:
            for store in self.kernel_ir.stores:
                lines.extend(self._indent_lines(self._emit_store(store), "    "))

        lines.append("    llvm.return")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def _format_signature(self) -> str:
        return ", ".join(
            f"%{arg.name}: {self._arg_type(arg)}" for arg in self.kernel_ir.args
        )

    def _arg_type(self, arg: KernelArg) -> str:
        if arg.spec.kind == "buffer":
            return "!llvm.ptr"
        if arg.spec.kind == "scalar":
            return self._scalar_type(arg.spec.dtype)
        raise MLIREmissionError(f"unsupported argument kind: {arg.spec.kind}")

    def _scalar_type(self, dtype: str) -> str:
        if dtype == "float32":
            return "f32"
        if dtype == "index":
            return "i64"
        if dtype == "int32":
            return "i32"
        raise MLIREmissionError(f"unsupported dtype: {dtype}")

    def _emit_value(self, value: ValueNode) -> str:
        if value.op == "program_id":
            return self._emit_program_id(value)
        if value.op == "arange":
            return self._emit_arange(value)
        if value.op == "cmp_lt":
            return self._emit_cmp_lt(value)
        if value.op == "load":
            return self._emit_load(value)
        if value.op == "add":
            return self._emit_add(value)
        raise MLIREmissionError(f"unsupported IR op: {value.op}")

    def _emit_arange(self, value: ValueNode) -> str:
        start = int(self._require_attr(value, "start"))
        end = int(self._require_attr(value, "end"))
        if end <= start:
            raise MLIREmissionError(f"invalid arange bounds: start={start}, end={end}")

        base_name = f"{value.name}_base"
        start_const_name = f"{value.name}_start"
        return "\n".join(
            [
                self._emit_program_id(
                    ValueNode(name=base_name, op="program_id", dtype="index", attrs=(("axis", "0"), ("scope", "'global'")))
                ),
                f"%{start_const_name} = llvm.mlir.constant({start} : i64) : i64",
                f"%{value.name} = llvm.add %{base_name}, %{start_const_name} : i64",
            ]
        )

    def _emit_program_id(self, value: ValueNode) -> str:
        axis = self._require_attr(value, "axis")
        scope = self._require_attr(value, "scope")
        if axis != "0":
            raise MLIREmissionError(f"unsupported program_id axis: {axis}")
        if scope != "'global'":
            raise MLIREmissionError(f"unsupported program_id scope: {scope}")

        block_id_name = f"{value.name}_block_id"
        block_dim_name = f"{value.name}_block_dim"
        thread_id_name = f"{value.name}_thread_id"
        block_id_i64_name = f"{block_id_name}_i64"
        block_dim_i64_name = f"{block_dim_name}_i64"
        thread_id_i64_name = f"{thread_id_name}_i64"
        block_base_name = f"{value.name}_block_base"
        return "\n".join(
            [
                f"%{block_id_name} = nvvm.read.ptx.sreg.ctaid.x : i32",
                f"%{block_dim_name} = nvvm.read.ptx.sreg.ntid.x : i32",
                f"%{thread_id_name} = nvvm.read.ptx.sreg.tid.x : i32",
                f"%{block_id_i64_name} = llvm.sext %{block_id_name} : i32 to i64",
                f"%{block_dim_i64_name} = llvm.sext %{block_dim_name} : i32 to i64",
                f"%{thread_id_i64_name} = llvm.sext %{thread_id_name} : i32 to i64",
                f"%{block_base_name} = llvm.mul %{block_id_i64_name}, %{block_dim_i64_name} : i64",
                f"%{value.name} = llvm.add %{block_base_name}, %{thread_id_i64_name} : i64",
            ]
        )

    def _emit_cmp_lt(self, value: ValueNode) -> str:
        lhs, rhs = value.inputs
        return f"%{value.name} = llvm.icmp \"slt\" %{lhs}, %{rhs} : i64"

    def _emit_load(self, value: ValueNode) -> str:
        buffer_name = value.inputs[0]
        index_name = value.inputs[1]
        buffer_spec = self._lookup_buffer_spec(buffer_name)
        ptr_name = f"{value.name}_ptr"
        return (
            f"%{ptr_name} = llvm.getelementptr %{buffer_name}[%{index_name}] : "
            f"(!llvm.ptr, i64) -> !llvm.ptr, {self._scalar_type(buffer_spec.dtype)}\n"
            f"%{value.name} = llvm.load %{ptr_name} : !llvm.ptr -> {self._scalar_type(buffer_spec.dtype)}"
        )

    def _emit_add(self, value: ValueNode) -> str:
        lhs, rhs = value.inputs
        if value.dtype == "float32":
            return f"%{value.name} = llvm.fadd %{lhs}, %{rhs} : f32"
        if value.dtype == "int32":
            return f"%{value.name} = llvm.add %{lhs}, %{rhs} : i32"
        raise MLIREmissionError(f"unsupported add dtype: {value.dtype}")

    def _emit_store(self, store: StoreNode) -> str:
        buffer_spec = self._lookup_buffer_spec(store.buffer)
        ptr_name = f"{store.value}_store_ptr"
        return (
            f"%{ptr_name} = llvm.getelementptr %{store.buffer}[%{store.index}] : "
            f"(!llvm.ptr, i64) -> !llvm.ptr, {self._scalar_type(buffer_spec.dtype)}\n"
            f"llvm.store %{store.value}, %{ptr_name} : {self._scalar_type(buffer_spec.dtype)}, !llvm.ptr"
        )

    def _lookup_buffer_spec(self, arg_name: str) -> ArgSpec:
        if arg_name not in self._arg_specs:
            raise MLIREmissionError(f"unknown argument referenced in IR: {arg_name}")
        spec = self._arg_specs[arg_name]
        if spec.kind != "buffer":
            raise MLIREmissionError(f"expected buffer argument, got {spec.kind}: {arg_name}")
        return spec

    def _require_attr(self, value: ValueNode, attr_name: str) -> str:
        for key, raw_value in value.attrs:
            if key == attr_name:
                return raw_value
        raise MLIREmissionError(f"missing required attribute '{attr_name}' on {value.name}")

    def _determine_common_mask(self) -> str | None:
        masks = {store.mask for store in self.kernel_ir.stores if store.mask is not None}
        if not masks:
            return None
        if len(masks) != 1:
            raise MLIREmissionError("multiple distinct store masks are not supported yet")

        mask_name = next(iter(masks))
        for value in self.kernel_ir.values:
            if value.op == "load" and len(value.inputs) == 3 and value.inputs[2] != mask_name:
                raise MLIREmissionError("multiple distinct load masks are not supported yet")
        return mask_name

    def _determine_guarded_values(self, guard_name: str | None) -> set[str]:
        if guard_name is None:
            return set()

        guarded = {
            value.name
            for value in self.kernel_ir.values
            if value.op == "load" and len(value.inputs) == 3 and value.inputs[2] == guard_name
        }

        changed = True
        while changed:
            changed = False
            for value in self.kernel_ir.values:
                if value.name in guarded:
                    continue
                if any(input_name in guarded for input_name in value.inputs):
                    guarded.add(value.name)
                    changed = True

        return guarded

    def _indent_lines(self, text: str, indent: str) -> list[str]:
        return [f"{indent}{line}" for line in text.splitlines()]