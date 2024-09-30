set(LLVM_TARGET_DEFINITIONS vLLMGraphOps.td)
mlir_tablegen(vLLMGraphOps.h.inc -gen-op-decls)
mlir_tablegen(vLLMGraphOps.cpp.inc -gen-op-defs)
mlir_tablegen(vLLMGraphIRDialect.h.inc -gen-dialect-decls -dialect=vllm_graph)
mlir_tablegen(vLLMGraphIRDialect.cpp.inc -gen-dialect-defs -dialect=vllm_graph)
add_public_tablegen_target(MLIRvLLMGraphOpsIncGen)
add_dependencies(mlir-headers MLIRvLLMGraphOpsIncGen)


add_mlir_doc(vLLMGraphIRDialect vLLMGraphIRDialect vLLMGraph/ -gen-dialect-doc)
add_mlir_doc(vLLMGraphOps vLLMGraphOps vLLMGraph/ -gen-op-doc)
