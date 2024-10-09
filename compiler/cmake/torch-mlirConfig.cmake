# torch-mlirConfig.cmake
set(torch-mlir_INCLUDE_DIR ${TORCH_MLIR_DIR}/include)
set(torch-mlir_LIBRARY ${TORCH_MLIR_DIR}/lib)

# Define a target for the library if necessary
add_library(torch-mlir UNKNOWN IMPORTED)
set_target_properties(torch-mlir PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${torch-mlir_INCLUDE_DIR}"
    IMPORTED_LOCATION "${torch-mlir_LIBRARY}"
)
