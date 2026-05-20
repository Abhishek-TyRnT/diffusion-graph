
add_mlir_library(DiffusionGraphCompiler
    Interface.cpp

    LINK_LIBS PUBLIC
    DiffusionGraphInitAll
    MLIRPass
    MLIRSparseTensorDialect
    DiffusionGraphTransformPasses
    TorchMLIRTorchUtils
    MLIRSparseTensorUtils
    TorchMLIRTorchDialect
    )