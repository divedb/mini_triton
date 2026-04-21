from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from mini_triton import (
    DevicePointer,
    KernelArgumentError,
    LaunchConfig,
    RuntimeUnavailableError,
    buffer,
    load_kernel,
    launch_kernel,
    marshal_kernel_arguments,
    scalar,
)


def test_launch_config_rounds_up_grid_size():
    config = LaunchConfig.for_num_elements(257, 128)

    assert config.grid_x == 3
    assert config.block_x == 128
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


def test_load_kernel_requires_driver_when_cuda_binding_is_unavailable(tmp_path: Path):
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(".visible .entry add_kernel() { ret; }\n", encoding="utf-8")

    with pytest.raises(RuntimeUnavailableError, match="kernel driver implementation is required"):
        load_kernel(ptx_path, "add_kernel")


def test_load_and_launch_kernel_with_injected_driver(tmp_path: Path):
    ptx_text = ".visible .entry add_kernel() { ret; }\n"
    ptx_path = tmp_path / "kernel.ptx"
    ptx_path.write_text(ptx_text, encoding="utf-8")
    driver = _FakeKernelDriver()

    kernel = load_kernel(ptx_path, "add_kernel", driver=driver)
    config = LaunchConfig.for_num_elements(65, 64)
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
            "block_x": 64,
            "shared_mem_bytes": 0,
            "arg_names": ("x", "y", "out", "n"),
            "scalar_n": 65,
        }
    ]
    assert marshaled.names == ("x", "y", "out", "n")


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
                "block_x": launch_config.block_x,
                "shared_mem_bytes": launch_config.shared_mem_bytes,
                "arg_names": arguments.names,
                "scalar_n": arguments.storage[-1].value,
            }
        )