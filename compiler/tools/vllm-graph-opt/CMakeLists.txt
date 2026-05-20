add_llvm_executable(diffusion-graph-opt diffusion-graph-opt.cpp)

install(TARGETS diffusion-graph-opt
EXPORT vLLLMGraphTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT diffusion-graph-opt)

set(dependency_libraries)

target_link_libraries(diffusion-graph-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  DiffusionGraphInitAll
  DiffusionGraphIRDialect
  DiffusionGraphConversionPasses
  TorchMLIRTorchDialect
  ${dependency_libraries}
)