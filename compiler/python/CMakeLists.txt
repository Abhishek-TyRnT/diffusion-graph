
pybind11_add_module(graph_compiler vLLMGraphBindings.cpp GraphConvertor/GraphWriter.cpp)
install(TARGETS graph_compiler DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
target_link_libraries(graph_compiler PRIVATE
    vLLMGraphCompiler
    MLIRCAPIIR
  )