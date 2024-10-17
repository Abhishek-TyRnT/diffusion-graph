

add_mlir_library(vLLMGraphIRDialect
  vLLMGraphDialect.cpp
  vLLMGraphOps.cpp
  vLLMGraphOpsODSGenerated.cpp
  vLLMGraphTypes.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/vllm_graph/Dialect

  DEPENDS
  MLIRvLLMGraphOpsIncGen
  MLIRvLLMGraphTypesIncGen
  

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRFuncDialect
  MLIRIR
  MLIRArithDialect
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

#torch_mlir_target_includes(vLLMGraphIRDialect)
