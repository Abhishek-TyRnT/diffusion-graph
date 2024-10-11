cmake_minimum_required(VERSION 3.12)

project(vllm-graph-compiler)


find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(TORCH_MLIR_BUILD_DIR ${CMAKE_BINARY_DIR}/external/torch-mlir)
add_subdirectory(${TORCH_MLIR_SOURCE_DIR} ${TORCH_MLIR_BUILD_DIR})
set(CMAKE_CXX_STANDARD 17)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TORCH_MLIR_SOURCE_DIR}/include)
include_directories(${TORCH_MLIR_BUILD_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(include)
add_subdirectory(lib)

add_llvm_executable(vllm-graph-opt vllm-graph-opt.cpp)



install(TARGETS vllm-graph-opt
EXPORT vLLLMGraphTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT vllm-graph-opt)

set(dependency_libraries)

target_link_libraries(vllm-graph-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  vLLMGraphInitAll
  vLLMGraphIRDialect
  vLLMGraphConversionPasses
  ${dependency_libraries}
)
