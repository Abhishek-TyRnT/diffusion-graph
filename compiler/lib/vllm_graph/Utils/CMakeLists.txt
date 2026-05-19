
add_mlir_library(vLLMGraphUtils
    Utils.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/vllm_graph/Utils

    LINK_LIBS PUBLIC
    MLIRIR
    DiffusionGraphIRDialect
    TorchMLIRTorchDialect
    MLIRTransforms
)