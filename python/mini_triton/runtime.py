from __future__ import annotations

import ctypes
import os
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


class CudaDriverError(MiniTritonRuntimeError):
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
    grid_y: int = 1
    grid_z: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_mem_bytes: int = 0
    stream: int | None = None

    def __post_init__(self) -> None:
        if self.grid_x <= 0:
            raise KernelArgumentError(f"grid_x must be positive, got {self.grid_x}")
        if self.grid_y <= 0:
            raise KernelArgumentError(f"grid_y must be positive, got {self.grid_y}")
        if self.grid_z <= 0:
            raise KernelArgumentError(f"grid_z must be positive, got {self.grid_z}")
        if self.block_x <= 0:
            raise KernelArgumentError(f"block_x must be positive, got {self.block_x}")
        if self.block_y <= 0:
            raise KernelArgumentError(f"block_y must be positive, got {self.block_y}")
        if self.block_z <= 0:
            raise KernelArgumentError(f"block_z must be positive, got {self.block_z}")
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


@dataclass
class _CudaModuleHandle:
    module: ctypes.c_void_p
    ptx_buffer: ctypes.Array[ctypes.c_char]


class CtypesCudaDriver:
    def __init__(self, library: object | None = None) -> None:
        self._driver = library if library is not None else _load_nvcuda_library()

        self._cuInit = self._driver.cuInit
        self._cuDeviceGet = self._driver.cuDeviceGet
        self._cuCtxCreate = self._resolve_ctx_create(self._driver)
        self._cuMemAlloc = self._resolve_symbol(self._driver, "cuMemAlloc_v2", "cuMemAlloc")
        self._cuMemFree = self._resolve_symbol(self._driver, "cuMemFree_v2", "cuMemFree")
        self._cuMemcpyHtoD = self._resolve_symbol(self._driver, "cuMemcpyHtoD_v2", "cuMemcpyHtoD")
        self._cuMemcpyDtoH = self._resolve_symbol(self._driver, "cuMemcpyDtoH_v2", "cuMemcpyDtoH")
        self._cuCtxSynchronize = getattr(self._driver, "cuCtxSynchronize", None)
        self._cuModuleLoadDataEx = self._driver.cuModuleLoadDataEx
        self._cuModuleGetFunction = self._driver.cuModuleGetFunction
        self._cuLaunchKernel = self._driver.cuLaunchKernel
        self._cuGetErrorName = getattr(self._driver, "cuGetErrorName", None)

        _configure_ctypes_signature(self._cuInit, ctypes.c_int, [ctypes.c_uint])
        _configure_ctypes_signature(self._cuDeviceGet, ctypes.c_int, [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        _configure_ctypes_signature(
            self._cuCtxCreate,
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
        )
        _configure_ctypes_signature(
            self._cuMemAlloc,
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
        )
        _configure_ctypes_signature(
            self._cuMemFree,
            ctypes.c_int,
            [ctypes.c_uint64],
        )
        _configure_ctypes_signature(
            self._cuMemcpyHtoD,
            ctypes.c_int,
            [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
        )
        _configure_ctypes_signature(
            self._cuMemcpyDtoH,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        )
        if self._cuCtxSynchronize is not None:
            _configure_ctypes_signature(self._cuCtxSynchronize, ctypes.c_int, [])
        _configure_ctypes_signature(
            self._cuModuleLoadDataEx,
            ctypes.c_int,
            [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        _configure_ctypes_signature(
            self._cuModuleGetFunction,
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
        )
        _configure_ctypes_signature(
            self._cuLaunchKernel,
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        if self._cuGetErrorName is not None:
            _configure_ctypes_signature(
                self._cuGetErrorName,
                ctypes.c_int,
                [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
            )

        self._check(self._cuInit(0), "cuInit")
        self._context = self._create_primary_context()

    def _resolve_ctx_create(self, driver: object) -> object:
        for name in ("cuCtxCreate_v2", "cuCtxCreate"):
            candidate = getattr(driver, name, None)
            if candidate is not None:
                return candidate
        raise RuntimeUnavailableError("CUDA driver does not expose cuCtxCreate entrypoint")

    def _resolve_symbol(self, driver: object, *names: str) -> object:
        for name in names:
            candidate = getattr(driver, name, None)
            if candidate is not None:
                return candidate
        joined = ", ".join(names)
        raise RuntimeUnavailableError(f"CUDA driver does not expose required symbol(s): {joined}")

    def _create_primary_context(self) -> ctypes.c_void_p:
        device = ctypes.c_int()
        self._check(self._cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

        context = ctypes.c_void_p()
        self._check(self._cuCtxCreate(ctypes.byref(context), 0, device.value), "cuCtxCreate")
        return context

    def load_module(self, ptx_text: str) -> object:
        module = ctypes.c_void_p()
        ptx_bytes = ptx_text.encode("utf-8") + b"\0"
        ptx_buffer = ctypes.create_string_buffer(ptx_bytes)
        image_ptr = ctypes.cast(ptx_buffer, ctypes.c_void_p)

        self._check(
            self._cuModuleLoadDataEx(
                ctypes.byref(module),
                image_ptr,
                0,
                None,
                None,
            ),
            "cuModuleLoadDataEx",
        )
        return _CudaModuleHandle(module=module, ptx_buffer=ptx_buffer)

    def get_function(self, module_handle: object, kernel_name: str) -> object:
        if not isinstance(module_handle, _CudaModuleHandle):
            raise CudaDriverError("unexpected module handle type")

        function = ctypes.c_void_p()
        self._check(
            self._cuModuleGetFunction(
                ctypes.byref(function),
                module_handle.module,
                kernel_name.encode("utf-8"),
            ),
            "cuModuleGetFunction",
        )
        return function

    def launch_kernel(
        self,
        function_handle: object,
        *,
        launch_config: LaunchConfig,
        arguments: MarshaledKernelArguments,
    ) -> None:
        if not isinstance(function_handle, ctypes.c_void_p):
            raise CudaDriverError("unexpected function handle type")

        kernel_params_ptr = ctypes.cast(arguments.argument_pointers, ctypes.POINTER(ctypes.c_void_p))
        stream_handle = ctypes.c_void_p(0 if launch_config.stream is None else launch_config.stream)

        self._check(
            self._cuLaunchKernel(
                function_handle,
                launch_config.grid_x,
                launch_config.grid_y,
                launch_config.grid_z,
                launch_config.block_x,
                launch_config.block_y,
                launch_config.block_z,
                launch_config.shared_mem_bytes,
                stream_handle,
                kernel_params_ptr,
                None,
            ),
            "cuLaunchKernel",
        )

    def mem_alloc(self, nbytes: int) -> DevicePointer:
        if nbytes <= 0:
            raise CudaDriverError(f"allocation size must be positive, got {nbytes}")

        device_ptr = ctypes.c_uint64()
        self._check(self._cuMemAlloc(ctypes.byref(device_ptr), nbytes), "cuMemAlloc")
        return DevicePointer(int(device_ptr.value))

    def mem_free(self, pointer: DevicePointer | int) -> None:
        address = _coerce_device_pointer(pointer).address
        self._check(self._cuMemFree(ctypes.c_uint64(address)), "cuMemFree")

    def memcpy_htod(self, dst: DevicePointer | int, src: bytes | bytearray | memoryview) -> None:
        address = _coerce_device_pointer(dst).address
        src_view = memoryview(src)
        data = src_view.tobytes()
        src_buffer = ctypes.create_string_buffer(data)
        self._check(
            self._cuMemcpyHtoD(
                ctypes.c_uint64(address),
                ctypes.cast(src_buffer, ctypes.c_void_p),
                len(data),
            ),
            "cuMemcpyHtoD",
        )

    def memcpy_dtoh(self, src: DevicePointer | int, nbytes: int) -> bytes:
        if nbytes < 0:
            raise CudaDriverError(f"copy size must be non-negative, got {nbytes}")

        address = _coerce_device_pointer(src).address
        out_buffer = ctypes.create_string_buffer(nbytes)
        self._check(
            self._cuMemcpyDtoH(
                ctypes.cast(out_buffer, ctypes.c_void_p),
                ctypes.c_uint64(address),
                nbytes,
            ),
            "cuMemcpyDtoH",
        )
        return out_buffer.raw

    def synchronize(self) -> None:
        if self._cuCtxSynchronize is None:
            return
        self._check(self._cuCtxSynchronize(), "cuCtxSynchronize")

    def _check(self, code: int, api_name: str) -> None:
        if code == 0:
            return
        raise CudaDriverError(f"{api_name} failed with code {code} ({self._error_name(code)})")

    def _error_name(self, code: int) -> str:
        if self._cuGetErrorName is None:
            return "unknown"
        error_name = ctypes.c_char_p()
        result = self._cuGetErrorName(code, ctypes.byref(error_name))
        if result != 0 or not error_name.value:
            return "unknown"
        return error_name.value.decode("utf-8", errors="replace")


def _configure_ctypes_signature(func: object, restype: object, argtypes: list[object]) -> None:
    try:
        setattr(func, "restype", restype)
        setattr(func, "argtypes", argtypes)
    except (AttributeError, TypeError):
        # Allow pure-Python test doubles that do not expose ctypes function attributes.
        return


def _load_nvcuda_library() -> object:
    if os.name == "nt":
        try:
            return ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise RuntimeUnavailableError(
                "nvcuda.dll could not be loaded; install an NVIDIA driver or inject a custom KernelDriver"
            ) from exc

    for candidate in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise RuntimeUnavailableError(
        "CUDA driver library could not be loaded; install CUDA driver or inject a custom KernelDriver"
    )


def load_kernel(
    ptx_path: str | Path,
    kernel_name: str,
    *,
    driver: KernelDriver | None = None,
) -> LoadedKernel:
    if driver is None:
        driver = CtypesCudaDriver()

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