add_llvm_executable(vllm-graph-opt vllm-graph-opt.cpp)

install(TARGETS vllm-graph-opt
EXPORT vLLLMGraphTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT vllm-graph-opt)

set(dependency_libraries)

target_link_libraries(vllm-graph-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  vLLMGraphInitAll
  vLLMGraphIRDialect
  vLLMGraphConversionPasses
  TorchMLIRTorchDialect
  ${dependency_libraries}
)