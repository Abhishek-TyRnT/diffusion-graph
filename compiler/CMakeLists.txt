cmake_minimum_required(VERSION 3.12)

project(vllm-graph-compiler)

list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_PATH})
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Find the HDF5 library
find_package(HDF5 REQUIRED COMPONENTS CXX)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/interface)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories("/usr/include/python3.10")
include_directories(${HDF5_INCLUDE_DIRS})


list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(interface)
add_subdirectory(tools)
add_subdirectory(python)

# pybind11_add_module(graph_compiler python/vLLMGraphBindings.cpp)
# install(TARGETS graph_compiler DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
# target_link_libraries(graph_compiler PRIVATE 
#   vLLMGraphInitAll
#   MLIRCAPIIR)

