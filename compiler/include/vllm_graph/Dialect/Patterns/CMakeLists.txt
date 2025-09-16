
#TODO: Currently creating only symlinks, but need to copy it when packaging
set(PYTHON_FILES
  ${CMAKE_SOURCE_DIR}/include/vllm_graph/Dialect/Patterns/poolingLayer.pdl
)

foreach(python_file IN LISTS PYTHON_FILES)
  get_filename_component(filename ${python_file} NAME)
  add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/patterns/${filename}
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${python_file} ${CMAKE_BINARY_DIR}/patterns/${filename}
      DEPENDS ${python_file}
      COMMENT "Creating symlink for ${filename}"
  )
  list(APPEND SYMLINK_TARGETS ${CMAKE_BINARY_DIR}/patterns/${filename})
endforeach()

add_custom_target(create_dialect_pattern_symlinks ALL DEPENDS ${SYMLINK_TARGETS})
