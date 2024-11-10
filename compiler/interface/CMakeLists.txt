
add_mlir_library(vLLMGraphCompiler
    Interface.cpp

    LINK_LIBS PUBLIC
    vLLMGraphInitAll
    MLIRPass
    MLIRSparseTensorDialect
    vLLMGraphTransformPasses
    TorchMLIRTorchUtils
    MLIRSparseTensorUtils
    TorchMLIRTorchDialect
    )