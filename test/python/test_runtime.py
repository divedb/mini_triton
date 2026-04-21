from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from mini_triton import (
    DevicePointer,
    KernelArgumentError,
    LaunchConfig,
    CtypesCudaDriver,
    CudaDriverError,
    RuntimeUnavailableError,
    buffer,
    clear_loaded_kernel_cache,
    load_kernel,
    launch_kernel,
    marshal_kernel_arguments,
    scalar,
)


def test_launch_config_rounds_up_grid_size():
    config = LaunchConfig.for_num_elements(257, 128)

    assert config.grid_x == 3
    assert config.grid_y == 1
    assert config.grid_z == 1
    assert config.block_x == 128
    assert config.block_y == 1
    assert config.block_z == 1
    assert config.shared_mem_bytes == 0
    assert config.stream is None


def test_marshal_kernel_arguments_preserves_declared_order():
    arg_specs = {
        "x": buffer("float32"),
        "y": buffer("float32"),
        "out": buffer("float32"),
        "n": scalar("index"),
    }

    marshaled = marshal_kernel_arguments(
        arg_specs,
        {
            "x": DevicePointer(0x1000),
            "y": 0x2000,
            "out": 0x3000,
            "n": 37,
        },
    )

    assert marshaled.names == ("x", "y", "out", "n")
    assert [value.value for value in marshaled.storage] == [0x1000, 0x2000, 0x3000, 37]
    assert len(marshaled.pointer_values()) == 4
    assert all(pointer > 0 for pointer in marshaled.pointer_values())
    assert isinstance(marshaled.storage[0], ctypes.c_void_p)
    assert isinstance(marshaled.storage[3], ctypes.c_int64)


def test_marshal_kernel_arguments_requires_declared_order():
    arg_specs = {
        "x": buffer("float32"),
        "n": scalar("index"),
    }

    with pytest.raises(KernelArgumentError, match="declared order"):
        marshal_kernel_arguments(
            arg_specs,
            {
                "n": 4,
                "x": 0x1000,
            },
        )


def test_load_kernel_requires_driver_when_cuda_binding_is_unavailable(tmp_path: Path, monkeypatch):
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(".visible .entry add_kernel() { ret; }\n", encoding="utf-8")

    from mini_triton import runtime

    def _raise_unavailable():
        raise RuntimeUnavailableError("not available")

    monkeypatch.setattr(runtime, "_load_nvcuda_library", _raise_unavailable)

    with pytest.raises(RuntimeUnavailableError, match="not available"):
        load_kernel(ptx_path, "add_kernel")


def test_load_and_launch_kernel_with_injected_driver(tmp_path: Path):
    ptx_text = ".visible .entry add_kernel() { ret; }\n"
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    driver = _FakeKernelDriver()

    kernel = load_kernel(ptx_path, "add_kernel", driver=driver)
    config = LaunchConfig(grid_x=2, grid_y=3, block_x=64, block_y=2)
    arg_specs = {
        "x": buffer("float32"),
        "y": buffer("float32"),
        "out": buffer("float32"),
        "n": scalar("index"),
    }

    marshaled = launch_kernel(
        kernel,
        arg_specs=arg_specs,
        args={
            "x": 0x1000,
            "y": 0x2000,
            "out": 0x3000,
            "n": 65,
        },
        launch_config=config,
    )

    assert kernel.ptx_text == ptx_text
    assert driver.loaded_modules == [ptx_text]
    assert driver.requested_functions == [("module-1", "add_kernel")]
    assert driver.launches == [
        {
            "function_handle": "function:add_kernel",
            "grid_x": 2,
            "grid_y": 3,
            "grid_z": 1,
            "block_x": 64,
            "block_y": 2,
            "block_z": 1,
            "shared_mem_bytes": 0,
            "arg_names": ("x", "y", "out", "n"),
            "scalar_n": 65,
        }
    ]
    assert marshaled.names == ("x", "y", "out", "n")


def test_load_kernel_reuses_cached_module_for_same_driver(tmp_path: Path):
    ptx_text = ".visible .entry add_kernel() { ret; }\n"
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    driver = _FakeKernelDriver()
    clear_loaded_kernel_cache()

    first = load_kernel(ptx_path, "add_kernel", driver=driver)
    second = load_kernel(ptx_path, "add_kernel", driver=driver)

    assert first is second
    assert driver.loaded_modules == [ptx_text]
    assert driver.requested_functions == [("module-1", "add_kernel")]


def test_load_kernel_cache_invalidates_when_ptx_changes(tmp_path: Path):
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(".visible .entry add_kernel() { ret; }\n", encoding="utf-8")
    driver = _FakeKernelDriver()
    clear_loaded_kernel_cache()

    first = load_kernel(ptx_path, "add_kernel", driver=driver)
    ptx_path.write_text("// changed\n.visible .entry add_kernel() { ret; }\n", encoding="utf-8")
    second = load_kernel(ptx_path, "add_kernel", driver=driver)

    assert first is not second
    assert len(driver.loaded_modules) == 2
    assert len(driver.requested_functions) == 2


def test_load_kernel_can_bypass_cache(tmp_path: Path):
    ptx_text = ".visible .entry add_kernel() { ret; }\n"
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    driver = _FakeKernelDriver()
    clear_loaded_kernel_cache()

    first = load_kernel(ptx_path, "add_kernel", driver=driver, use_cache=False)
    second = load_kernel(ptx_path, "add_kernel", driver=driver, use_cache=False)

    assert first is not second
    assert driver.loaded_modules == [ptx_text, ptx_text]
    assert driver.requested_functions == [
        ("module-1", "add_kernel"),
        ("module-1", "add_kernel"),
    ]


class _FakeKernelDriver:
    def __init__(self) -> None:
        self.loaded_modules: list[str] = []
        self.requested_functions: list[tuple[str, str]] = []
        self.launches: list[dict[str, object]] = []

    def load_module(self, ptx_text: str) -> object:
        self.loaded_modules.append(ptx_text)
        return "module-1"

    def get_function(self, module_handle: object, kernel_name: str) -> object:
        self.requested_functions.append((str(module_handle), kernel_name))
        return f"function:{kernel_name}"

    def launch_kernel(self, function_handle: object, *, launch_config: LaunchConfig, arguments) -> None:
        self.launches.append(
            {
                "function_handle": str(function_handle),
                "grid_x": launch_config.grid_x,
                "grid_y": launch_config.grid_y,
                "grid_z": launch_config.grid_z,
                "block_x": launch_config.block_x,
                "block_y": launch_config.block_y,
                "block_z": launch_config.block_z,
                "shared_mem_bytes": launch_config.shared_mem_bytes,
                "arg_names": arguments.names,
                "scalar_n": arguments.storage[-1].value,
            }
        )


def test_ctypes_cuda_driver_load_and_launch_with_fake_cuda_library():
    fake_library = _FakeCudaLibrary()
    driver = CtypesCudaDriver(library=fake_library)

    module = driver.load_module(".visible .entry add_kernel() { ret; }\n")
    function = driver.get_function(module, "add_kernel")
    marshaled = marshal_kernel_arguments(
        {
            "x": buffer("float32"),
            "y": buffer("float32"),
            "out": buffer("float32"),
            "n": scalar("index"),
        },
        {"x": 0x1000, "y": 0x2000, "out": 0x3000, "n": 33},
    )
    driver.launch_kernel(function, launch_config=LaunchConfig.for_num_elements(33, 32), arguments=marshaled)

    assert fake_library.calls[0] == "cuInit"
    assert "cuModuleLoadDataEx" in fake_library.calls
    assert "cuModuleGetFunction" in fake_library.calls
    assert "cuLaunchKernel" in fake_library.calls


def test_ctypes_cuda_driver_raises_error_name_from_cuda():
    fake_library = _FakeCudaLibrary(module_get_function_status=201)
    driver = CtypesCudaDriver(library=fake_library)
    module = driver.load_module(".visible .entry add_kernel() { ret; }\n")

    with pytest.raises(CudaDriverError, match="CUDA_ERROR_INVALID_CONTEXT"):
        driver.get_function(module, "add_kernel")


class _FakeCudaLibrary:
    def __init__(
        self,
        *,
        init_status: int = 0,
        module_load_status: int = 0,
        module_get_function_status: int = 0,
        launch_status: int = 0,
    ) -> None:
        self.init_status = init_status
        self.module_load_status = module_load_status
        self.module_get_function_status = module_get_function_status
        self.launch_status = launch_status
        self.calls: list[str] = []

        self.cuInit = self._cu_init
        self.cuDeviceGet = self._cu_device_get
        self.cuCtxCreate_v2 = self._cu_ctx_create
        self.cuMemAlloc_v2 = self._cu_mem_alloc
        self.cuMemFree_v2 = self._cu_mem_free
        self.cuMemcpyHtoD_v2 = self._cu_memcpy_htod
        self.cuMemcpyDtoH_v2 = self._cu_memcpy_dtoh
        self.cuCtxSynchronize = self._cu_ctx_synchronize
        self.cuModuleLoadDataEx = self._cu_module_load_data_ex
        self.cuModuleGetFunction = self._cu_module_get_function
        self.cuLaunchKernel = self._cu_launch_kernel
        self.cuGetErrorName = self._cu_get_error_name
        self._memory: dict[int, bytes] = {}
        self._next_address = 0x100000

    def _cu_init(self, _flags):
        self.calls.append("cuInit")
        return self.init_status

    def _cu_device_get(self, device_ptr, _ordinal):
        self.calls.append("cuDeviceGet")
        ctypes.cast(device_ptr, ctypes.POINTER(ctypes.c_int))[0] = ctypes.c_int(0)
        return 0

    def _cu_ctx_create(self, context_ptr, _flags, _device):
        self.calls.append("cuCtxCreate")
        ctypes.cast(context_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(0xCAFE)
        return 0

    def _cu_module_load_data_ex(self, module_ptr, _image, _num_options, _options, _option_values):
        self.calls.append("cuModuleLoadDataEx")
        if self.module_load_status == 0:
            ctypes.cast(module_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(0xAA01)
        return self.module_load_status

    def _cu_mem_alloc(self, ptr, size):
        self.calls.append("cuMemAlloc")
        address = self._next_address
        self._next_address += max(int(size), 1)
        self._memory[address] = b"\x00" * int(size)
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint64))[0] = ctypes.c_uint64(address)
        return 0

    def _cu_mem_free(self, ptr):
        self.calls.append("cuMemFree")
        self._memory.pop(int(ptr), None)
        return 0

    def _cu_memcpy_htod(self, dst, src, size):
        self.calls.append("cuMemcpyHtoD")
        data = ctypes.string_at(src, int(size))
        self._memory[int(dst)] = data
        return 0

    def _cu_memcpy_dtoh(self, dst, src, size):
        self.calls.append("cuMemcpyDtoH")
        data = self._memory.get(int(src), b"\x00" * int(size))
        ctypes.memmove(dst, data, int(size))
        return 0

    def _cu_ctx_synchronize(self):
        self.calls.append("cuCtxSynchronize")
        return 0

    def _cu_module_get_function(self, function_ptr, _module, _name):
        self.calls.append("cuModuleGetFunction")
        if self.module_get_function_status == 0:
            ctypes.cast(function_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(0xBB02)
        return self.module_get_function_status

    def _cu_launch_kernel(
        self,
        _function,
        _grid_x,
        _grid_y,
        _grid_z,
        _block_x,
        _block_y,
        _block_z,
        _shared_mem,
        _stream,
        _kernel_params,
        _extra,
    ):
        self.calls.append("cuLaunchKernel")
        return self.launch_status

    def _cu_get_error_name(self, code, out_name):
        self.calls.append("cuGetErrorName")
        mapping = {
            201: b"CUDA_ERROR_INVALID_CONTEXT",
        }
        name = mapping.get(code, b"CUDA_ERROR_UNKNOWN")
        ctypes.cast(out_name, ctypes.POINTER(ctypes.c_char_p))[0] = ctypes.c_char_p(name)
        return 0