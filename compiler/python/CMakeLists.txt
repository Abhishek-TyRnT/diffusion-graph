
add_subdirectory(GraphConvertor)
pybind11_add_module(graph_compiler vLLMGraphBindings.cpp)
install(TARGETS graph_compiler DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
target_link_libraries(graph_compiler PRIVATE
    vLLMGraphCompiler
    MLIRCAPIIR
    ${HDF5_LIBRARIES}
    GraphWriter
  )

set(PYTHON_FILES
  ${CMAKE_SOURCE_DIR}/python/compiler.py
)

foreach(python_file IN LISTS PYTHON_FILES)
  get_filename_component(filename ${python_file} NAME)
  add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/python/${filename}
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${python_file} ${CMAKE_BINARY_DIR}/python/${filename}
      DEPENDS ${python_file}
      COMMENT "Creating symlink for ${filename}"
  )
  list(APPEND SYMLINK_TARGETS ${CMAKE_BINARY_DIR}/python/${filename})
endforeach()

add_custom_target(create_python_symlinks ALL DEPENDS ${SYMLINK_TARGETS})
