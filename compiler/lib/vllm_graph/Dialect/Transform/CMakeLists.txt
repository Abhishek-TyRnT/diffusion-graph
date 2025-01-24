add_mlir_library(vLLMGraphTransformPasses
  Passes.cpp
  TransformPasses.cpp
  recompositionPass.cpp
  InlineDialectResourcesDictPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Dialect/Transform

  DEPENDS
  vLLMGraphTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRRewrite
  TorchMLIRTorchDialect
  vLLMGraphIRDialect
  vLLMGraphTorchTovLLMGraph
)