add_mlir_library(DiffusionGraphTransformPasses
  Passes.cpp
  TransformPasses.cpp
  recompositionPass.cpp
  InlineDialectResourcesDictPass.cpp
  FunctionPartitionPass.cpp
  staticMaterializationPass.cpp
  contiguousInsertionPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Dialect/Transform

  DEPENDS
  DiffusionGraphTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRRewrite
  TorchMLIRTorchDialect
  DiffusionGraphIRDialect
  DiffusionGraphTorchToDiffusionGraph
  DiffusionGraphPatternInterpreter
  DiffusionGraphUtils
)