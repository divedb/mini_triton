cmake_minimum_required(VERSION 3.20)

project(mini_triton LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LLVMConfigHelpers)

set(MINITRITON_LLVM_PROJECT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project"
    CACHE PATH
    "Path to the local llvm-project source checkout")

set(MINITRITON_LLVM_BUILD_DIR
    ""
    CACHE PATH
    "Path to an existing LLVM/MLIR build directory")

set(MINITRITON_CUDA_ARCH
    "sm_80"
    CACHE STRING
    "Default CUDA architecture used for PTX generation")

option(MINITRITON_ENABLE_TOOLCHAIN_CHECK
       "Validate the configured LLVM/MLIR build tree and tools"
       OFF)

option(MINITRITON_BUILD_CPP_LOWERING_DRIVER
    "Build the C++ lowering driver used by Python to produce PTX"
    ON)

option(MINITRITON_ENABLE_TD_GENERATION
    "Generate artifacts from MiniTritonOps.td when mlir-tblgen is available"
    ON)

mini_triton_infer_llvm_cmake_dirs(
    "${MINITRITON_LLVM_BUILD_DIR}"
    MINITRITON_INFERRED_LLVM_DIR
    MINITRITON_INFERRED_MLIR_DIR)

if(NOT LLVM_DIR AND MINITRITON_INFERRED_LLVM_DIR)
  set(LLVM_DIR
      "${MINITRITON_INFERRED_LLVM_DIR}"
      CACHE PATH
      "Path to LLVMConfig.cmake" FORCE)
endif()

if(NOT MLIR_DIR AND MINITRITON_INFERRED_MLIR_DIR)
  set(MLIR_DIR
      "${MINITRITON_INFERRED_MLIR_DIR}"
      CACHE PATH
      "Path to MLIRConfig.cmake" FORCE)
endif()

message(STATUS "mini_triton configuration")
message(STATUS "  llvm-project source: ${MINITRITON_LLVM_PROJECT_DIR}")
message(STATUS "  LLVM/MLIR build dir: ${MINITRITON_LLVM_BUILD_DIR}")
message(STATUS "  LLVM_DIR: ${LLVM_DIR}")
message(STATUS "  MLIR_DIR: ${MLIR_DIR}")
message(STATUS "  CUDA arch: ${MINITRITON_CUDA_ARCH}")

if(MINITRITON_ENABLE_TOOLCHAIN_CHECK)
  mini_triton_validate_llvm_setup(
      LLVM_PROJECT_DIR "${MINITRITON_LLVM_PROJECT_DIR}"
      LLVM_BUILD_DIR "${MINITRITON_LLVM_BUILD_DIR}")
endif()

if(MINITRITON_BUILD_CPP_LOWERING_DRIVER)
    add_executable(mini_triton_lower
            tools/mtc-lower/main.cpp
            tools/mtc-lower/dialect_ops.cpp
            tools/mtc-lower/dialect_model.cpp
            tools/mtc-lower/dialect_parser.cpp
            tools/mtc-lower/dialect_verifier.cpp
            tools/mtc-lower/dialect_emitter.cpp)
    target_compile_features(mini_triton_lower PRIVATE cxx_std_17)

    if(MINITRITON_ENABLE_TD_GENERATION)
        find_program(MINITRITON_MLIR_TBLGEN
                NAMES mlir-tblgen
                PATHS
                "${MINITRITON_LLVM_BUILD_DIR}/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/Release/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/RelWithDebInfo/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/Debug/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/x64/Release/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/x64/RelWithDebInfo/bin"
                "${MINITRITON_LLVM_BUILD_DIR}/x64/Debug/bin")

        if(MINITRITON_MLIR_TBLGEN)
            set(MINITRITON_TD_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/mini_triton")
            set(MINITRITON_TD_DOC "${MINITRITON_TD_OUTPUT_DIR}/MiniTritonOps.md")

            add_custom_command(
                    OUTPUT "${MINITRITON_TD_DOC}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${MINITRITON_TD_OUTPUT_DIR}"
                    COMMAND ${MINITRITON_MLIR_TBLGEN}
                    --gen-op-doc
                    -I "${MINITRITON_LLVM_PROJECT_DIR}/mlir/include"
                    -I "${MINITRITON_LLVM_PROJECT_DIR}/llvm/include"
                    -I "${MINITRITON_LLVM_BUILD_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/mtc-lower/MiniTritonOps.td"
                    -o "${MINITRITON_TD_DOC}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/mtc-lower/MiniTritonOps.td"
                    COMMENT "Generating MiniTriton ops docs from TableGen"
                    VERBATIM)

            add_custom_target(mini_triton_ops_doc DEPENDS "${MINITRITON_TD_DOC}")
            add_dependencies(mini_triton_lower mini_triton_ops_doc)
            message(STATUS "  MiniTriton TD generation enabled: ${MINITRITON_TD_DOC}")
        else()
            message(WARNING "MINITRITON_ENABLE_TD_GENERATION is ON but mlir-tblgen was not found under ${MINITRITON_LLVM_BUILD_DIR}/bin")
        endif()
    endif()
endif()

add_custom_target(mini_triton_scaffold ALL
        COMMENT "mini_triton scaffold")