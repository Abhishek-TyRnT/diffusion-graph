
add_subdirectory(TorchTovLLMGraph)

set(linked_libs vLLMGraphTorchTovLLMGraph)
add_mlir_library(vLLMGraphConversionPasses
  Passes.cpp

  DEPENDS
  vLLMGraphConversionTorchTovLLMGraphPassIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)

