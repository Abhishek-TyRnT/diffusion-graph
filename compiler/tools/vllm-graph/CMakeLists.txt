add_llvm_executable(vllm-graph vllm-graph.cpp)

install(TARGETS vllm-graph
EXPORT vLLLMGraphTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT vllm-graph)

set(dependency_libraries)

target_link_libraries(vllm-graph PRIVATE
  vLLMGraphCompiler
  TorchMLIRTorchDialect
  ${dependency_libraries}
)