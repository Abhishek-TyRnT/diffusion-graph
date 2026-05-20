

add_mlir_library(DiffusionGraphIRDialect
  DiffusionGraphDialect.cpp
  DiffusionGraphOps.cpp
  DiffusionGraphOpsODSGenerated.cpp
  DiffusionGraphTypes.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/diffusion_graph/Dialect

  DEPENDS
  MLIRDiffusionGraphOpsIncGen
  MLIRDiffusionGraphTypesIncGen
  

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
