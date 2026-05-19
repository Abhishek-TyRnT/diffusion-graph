set(LLVM_TARGET_DEFINITIONS vLLMGraphOps.td)
mlir_tablegen(DiffusionGraphOps.h.inc -gen-op-decls)
mlir_tablegen(DiffusionGraphOps.cpp.inc -gen-op-defs)
mlir_tablegen(DiffusionGraphIRDialect.h.inc -gen-dialect-decls -dialect=diffusion_graph)
mlir_tablegen(DiffusionGraphIRDialect.cpp.inc -gen-dialect-defs -dialect=diffusion_graph)
add_public_tablegen_target(MLIRDiffusionGraphOpsIncGen)
add_dependencies(mlir-headers MLIRDiffusionGraphOpsIncGen)


set(LLVM_TARGET_DEFINITIONS vLLMGraphTypes.td)
mlir_tablegen(DiffusionGraphTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DiffusionGraphTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRDiffusionGraphTypesIncGen)

add_mlir_doc(DiffusionGraphIRDialect DiffusionGraphIRDialect DiffusionGraph/ -gen-dialect-doc)
add_mlir_doc(DiffusionGraphOps DiffusionGraphOps DiffusionGraph/ -gen-op-doc)
