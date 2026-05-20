add_mlir_conversion_library(DiffusionGraphTorchToDiffusionGraph
  TorchToDiffusionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Conversion

  DEPENDS
  DiffusionGraphConversionTorchToDiffusionGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  DiffusionGraphIRDialect
  TorchMLIRTorchDialect
  MLIRArithDialect
  DiffusionGraphUtils
)