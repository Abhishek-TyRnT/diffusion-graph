add_mlir_conversion_library(vLLMGraphTorchTovLLMGraph
  TorchTovLLMGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Conversion

  DEPENDS
  vLLMGraphConversionTorchTovLLMGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  DiffusionGraphIRDialect
  TorchMLIRTorchDialect
  MLIRArithDialect
  vLLMGraphUtils
)