"""Public package surface for the mini_triton Phase 2 frontend scaffold."""

from .api import buffer, kernel, scalar
from .cache import CachedCompilation, compile_with_cache
from .compiler import (
	CompilationPlan,
	CompilationRun,
	CompilerConfigError,
	execute_compilation_plan,
	resolve_toolchain,
)
from .capture import CaptureError
from .ir import ArgSpec, KernelIR
from .mlir import MLIREmissionError, emit_mlir
from .runtime import (
	CtypesCudaDriver,
	CudaDriverError,
	DevicePointer,
	KernelArgumentError,
	LaunchConfig,
	LoadedKernel,
	MarshaledKernelArguments,
	MiniTritonRuntimeError,
	RuntimeUnavailableError,
	launch_kernel,
	load_kernel,
	marshal_kernel_arguments,
)

__all__ = [
	"ArgSpec",
	"CaptureError",
	"CachedCompilation",
	"CompilationPlan",
	"CompilationRun",
	"CompilerConfigError",
	"CtypesCudaDriver",
	"CudaDriverError",
	"DevicePointer",
	"KernelIR",
	"KernelArgumentError",
	"LaunchConfig",
	"LoadedKernel",
	"MLIREmissionError",
	"MarshaledKernelArguments",
	"MiniTritonRuntimeError",
	"RuntimeUnavailableError",
	"__version__",
	"buffer",
	"compile_with_cache",
	"emit_mlir",
	"execute_compilation_plan",
	"kernel",
	"launch_kernel",
	"load_kernel",
	"marshal_kernel_arguments",
	"resolve_toolchain",
	"scalar",
]

__version__ = "0.1.0"