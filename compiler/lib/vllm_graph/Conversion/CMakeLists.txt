add_mlir_conversion_library(vLLMGraphTorchTovLLMGraph
  TorchTovLLMGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Conversion

  DEPENDS
  vLLMGraphConversionTorchTovLLMGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTosaDialect
  vLLMGraphIRDialect
)

torch_mlir_target_includes(TorchMLIRTorchToTosa)
