set(LLVM_TARGET_DEFINITIONS vLLMGraphOps.td)
mlir_tablegen(vLLMGraphOps.h.inc -gen-op-decls)
mlir_tablegen(vLLMGraphOps.cpp.inc -gen-op-defs)
mlir_tablegen(vLLMGraphIRDialect.h.inc -gen-dialect-decls -dialect=vllm_graph)
mlir_tablegen(vLLMGraphIRDialect.cpp.inc -gen-dialect-defs -dialect=vllm_graph)
add_public_tablegen_target(MLIRvLLMGraphOpsIncGen)
add_dependencies(mlir-headers MLIRvLLMGraphOpsIncGen)


set(LLVM_TARGET_DEFINITIONS vLLMGraphTypes.td)
mlir_tablegen(vLLMGraphTypes.h.inc -gen-typedef-decls)
mlir_tablegen(vLLMGraphTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRvLLMGraphTypesIncGen)

add_mlir_doc(vLLMGraphIRDialect vLLMGraphIRDialect vLLMGraph/ -gen-dialect-doc)
add_mlir_doc(vLLMGraphOps vLLMGraphOps vLLMGraph/ -gen-op-doc)
