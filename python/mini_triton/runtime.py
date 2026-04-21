from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .ir import ArgSpec


class MiniTritonRuntimeError(RuntimeError):
    pass


class KernelArgumentError(MiniTritonRuntimeError):
    pass


class RuntimeUnavailableError(MiniTritonRuntimeError):
    pass


@dataclass(frozen=True)
class DevicePointer:
    address: int

    def __post_init__(self) -> None:
        if not isinstance(self.address, int):
            raise KernelArgumentError(
                f"device pointers must be represented as integers, got {type(self.address).__name__}"
            )
        if self.address <= 0:
            raise KernelArgumentError(f"device pointer addresses must be positive, got {self.address}")


@dataclass(frozen=True)
class LaunchConfig:
    grid_x: int
    block_x: int
    shared_mem_bytes: int = 0
    stream: int | None = None

    def __post_init__(self) -> None:
        if self.grid_x <= 0:
            raise KernelArgumentError(f"grid_x must be positive, got {self.grid_x}")
        if self.block_x <= 0:
            raise KernelArgumentError(f"block_x must be positive, got {self.block_x}")
        if self.shared_mem_bytes < 0:
            raise KernelArgumentError(
                f"shared_mem_bytes must be non-negative, got {self.shared_mem_bytes}"
            )
        if self.stream is not None and self.stream < 0:
            raise KernelArgumentError(f"stream must be non-negative when provided, got {self.stream}")

    @classmethod
    def for_num_elements(
        cls,
        num_elements: int,
        block_x: int,
        *,
        shared_mem_bytes: int = 0,
        stream: int | None = None,
    ) -> "LaunchConfig":
        if num_elements <= 0:
            raise KernelArgumentError(f"num_elements must be positive, got {num_elements}")
        if block_x <= 0:
            raise KernelArgumentError(f"block_x must be positive, got {block_x}")
        grid_x = (num_elements + block_x - 1) // block_x
        return cls(
            grid_x=grid_x,
            block_x=block_x,
            shared_mem_bytes=shared_mem_bytes,
            stream=stream,
        )


@dataclass(frozen=True)
class MarshaledKernelArguments:
    names: tuple[str, ...]
    storage: tuple[object, ...]
    argument_pointers: ctypes.Array[ctypes.c_void_p]

    def pointer_values(self) -> tuple[int, ...]:
        return tuple(int(pointer) for pointer in self.argument_pointers)


class KernelDriver(Protocol):
    def load_module(self, ptx_text: str) -> object: ...

    def get_function(self, module_handle: object, kernel_name: str) -> object: ...

    def launch_kernel(
        self,
        function_handle: object,
        *,
        launch_config: LaunchConfig,
        arguments: MarshaledKernelArguments,
    ) -> None: ...


@dataclass(frozen=True)
class LoadedKernel:
    kernel_name: str
    ptx_path: Path
    ptx_text: str
    driver: KernelDriver
    module_handle: object
    function_handle: object


def load_kernel(
    ptx_path: str | Path,
    kernel_name: str,
    *,
    driver: KernelDriver | None = None,
) -> LoadedKernel:
    if driver is None:
        raise RuntimeUnavailableError(
            "a kernel driver implementation is required; CUDA binding integration is not available in this environment"
        )

    resolved_path = Path(ptx_path).expanduser().resolve()
    ptx_text = resolved_path.read_text(encoding="utf-8")
    module_handle = driver.load_module(ptx_text)
    function_handle = driver.get_function(module_handle, kernel_name)
    return LoadedKernel(
        kernel_name=kernel_name,
        ptx_path=resolved_path,
        ptx_text=ptx_text,
        driver=driver,
        module_handle=module_handle,
        function_handle=function_handle,
    )


def launch_kernel(
    kernel: LoadedKernel,
    *,
    arg_specs: Mapping[str, ArgSpec],
    args: Mapping[str, object],
    launch_config: LaunchConfig,
) -> MarshaledKernelArguments:
    marshaled = marshal_kernel_arguments(arg_specs, args)
    kernel.driver.launch_kernel(
        kernel.function_handle,
        launch_config=launch_config,
        arguments=marshaled,
    )
    return marshaled


def marshal_kernel_arguments(
    arg_specs: Mapping[str, ArgSpec],
    args: Mapping[str, object],
) -> MarshaledKernelArguments:
    expected_names = tuple(arg_specs.keys())
    provided_names = tuple(args.keys())
    if expected_names != provided_names:
        raise KernelArgumentError(
            f"kernel arguments must match the declared order: expected {list(expected_names)}, got {list(provided_names)}"
        )

    storage: list[object] = []
    pointer_values: list[int] = []
    for name in expected_names:
        spec = arg_specs[name]
        c_value = _marshal_argument(spec, args[name])
        storage.append(c_value)
        pointer_values.append(ctypes.addressof(c_value))

    argument_pointers = (ctypes.c_void_p * len(pointer_values))(
        *(ctypes.c_void_p(pointer) for pointer in pointer_values)
    )
    return MarshaledKernelArguments(
        names=expected_names,
        storage=tuple(storage),
        argument_pointers=argument_pointers,
    )


def _marshal_argument(spec: ArgSpec, value: object) -> object:
    if spec.kind == "buffer":
        return ctypes.c_void_p(_coerce_device_pointer(value).address)
    if spec.kind == "scalar":
        return _marshal_scalar(spec.dtype, value)
    raise KernelArgumentError(f"unsupported argument kind: {spec.kind}")


def _coerce_device_pointer(value: object) -> DevicePointer:
    if isinstance(value, DevicePointer):
        return value
    if isinstance(value, int):
        return DevicePointer(value)
    raise KernelArgumentError(
        f"buffer arguments must be device pointers or integer addresses, got {type(value).__name__}"
    )


def _marshal_scalar(dtype: str, value: object) -> object:
    if dtype == "index":
        if not isinstance(value, int):
            raise KernelArgumentError(f"scalar<{dtype}> expects int, got {type(value).__name__}")
        return ctypes.c_int64(value)
    if dtype == "int32":
        if not isinstance(value, int):
            raise KernelArgumentError(f"scalar<{dtype}> expects int, got {type(value).__name__}")
        return ctypes.c_int32(value)
    if dtype == "float32":
        if not isinstance(value, (int, float)):
            raise KernelArgumentError(f"scalar<{dtype}> expects float-compatible value, got {type(value).__name__}")
        return ctypes.c_float(value)
    raise KernelArgumentError(f"unsupported scalar dtype: {dtype}")