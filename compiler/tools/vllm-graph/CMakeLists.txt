add_llvm_executable(diffusion-graph diffusion-graph.cpp)

install(TARGETS diffusion-graph
EXPORT vLLLMGraphTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT diffusion-graph)

set(dependency_libraries)

target_link_libraries(diffusion-graph PRIVATE
  DiffusionGraphCompiler
  TorchMLIRTorchDialect
  ${dependency_libraries}
)