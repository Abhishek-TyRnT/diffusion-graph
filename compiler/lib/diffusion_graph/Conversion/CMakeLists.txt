
add_subdirectory(TorchToDiffusionGraph)

set(linked_libs DiffusionGraphTorchToDiffusionGraph)
add_mlir_library(DiffusionGraphConversionPasses
  Passes.cpp

  DEPENDS
  DiffusionGraphConversionTorchToDiffusionGraphPassIncGen

  LINK_LIBS PUBLIC
  ${linked_libs}
)

