function(mini_triton_infer_llvm_cmake_dirs llvm_build_dir out_llvm_dir out_mlir_dir)
  set(inferred_llvm_dir "")
  set(inferred_mlir_dir "")

  if(llvm_build_dir)
    cmake_path(NORMAL_PATH llvm_build_dir OUTPUT_VARIABLE normalized_build_dir)
    set(candidate_llvm_dir "${normalized_build_dir}/lib/cmake/llvm")
    set(candidate_mlir_dir "${normalized_build_dir}/lib/cmake/mlir")

    if(EXISTS "${candidate_llvm_dir}/LLVMConfig.cmake")
      set(inferred_llvm_dir "${candidate_llvm_dir}")
    endif()

    if(EXISTS "${candidate_mlir_dir}/MLIRConfig.cmake")
      set(inferred_mlir_dir "${candidate_mlir_dir}")
    endif()
  endif()

  set(${out_llvm_dir} "${inferred_llvm_dir}" PARENT_SCOPE)
  set(${out_mlir_dir} "${inferred_mlir_dir}" PARENT_SCOPE)
endfunction()

function(mini_triton_validate_llvm_setup)
  set(options)
  set(one_value_args LLVM_PROJECT_DIR LLVM_BUILD_DIR)
  set(multi_value_args)
  cmake_parse_arguments(MT "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT MT_LLVM_PROJECT_DIR)
    message(FATAL_ERROR "MINITRITON_LLVM_PROJECT_DIR must be set for toolchain validation")
  endif()

  if(NOT EXISTS "${MT_LLVM_PROJECT_DIR}/mlir")
    message(FATAL_ERROR
            "MINITRITON_LLVM_PROJECT_DIR does not look like an llvm-project checkout: ${MT_LLVM_PROJECT_DIR}")
  endif()

  if(NOT MT_LLVM_BUILD_DIR)
    message(FATAL_ERROR "MINITRITON_LLVM_BUILD_DIR must be set for toolchain validation")
  endif()

  if(NOT EXISTS "${MT_LLVM_BUILD_DIR}/bin")
    message(FATAL_ERROR
            "MINITRITON_LLVM_BUILD_DIR does not contain a bin directory: ${MT_LLVM_BUILD_DIR}")
  endif()

  mini_triton_infer_llvm_cmake_dirs(
      "${MT_LLVM_BUILD_DIR}"
      inferred_llvm_dir
      inferred_mlir_dir)

  if(NOT inferred_llvm_dir)
    message(FATAL_ERROR
            "Could not infer LLVM_DIR from MINITRITON_LLVM_BUILD_DIR=${MT_LLVM_BUILD_DIR}")
  endif()

  if(NOT inferred_mlir_dir)
    message(FATAL_ERROR
            "Could not infer MLIR_DIR from MINITRITON_LLVM_BUILD_DIR=${MT_LLVM_BUILD_DIR}")
  endif()

  foreach(tool_name IN ITEMS mlir-opt mlir-translate llc)
    find_program(tool_path
                 NAMES ${tool_name}
                 PATHS "${MT_LLVM_BUILD_DIR}/bin"
                 NO_DEFAULT_PATH)

    if(NOT tool_path)
      message(FATAL_ERROR
              "Required tool '${tool_name}' was not found under ${MT_LLVM_BUILD_DIR}/bin")
    endif()

    message(STATUS "  found ${tool_name}: ${tool_path}")
    unset(tool_path CACHE)
  endforeach()
endfunction()