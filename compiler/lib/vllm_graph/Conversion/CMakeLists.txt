add_mlir_conversion_library(vLLMGraphTorchTovLLMGraph
  TorchTovLLMGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Conversion

  DEPENDS
  vLLMGraphConversionTorchTovLLMGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  vLLMGraphIRDialect
)

set(linked_libs vLLMGraphTorchTovLLMGraph)
add_mlir_library(vLLMGraphConversionPasses
  Passes.cpp

  DEPENDS
  vLLMGraphConversionTorchTovLLMGraphPassIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)

