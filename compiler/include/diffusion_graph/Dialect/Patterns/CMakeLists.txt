
# Input PDLL file

set(PatternsDir ${CMAKE_SOURCE_DIR}/include/diffusion_graph/Dialect/Patterns)
set(PDLL_FILES ${PatternsDir}/poolingLayer.pdl.mlir
               ${PatternsDir}/CLIPPoolingLayer.pdl.mlir)

# Generated intermediate and final outputs
set(PDL_FILE "")
set(PDL_INTERP_FILE "")

set(MLIR_BIN_PATH "/llvm-build/llvm-project/build/bin")
# Step 1: Lower PDLL -> PDL
foreach(PDLL_FILE ${PDLL_FILES})
  message(${MLIR_INCLUDE_DIRS})
  get_filename_component(NAME_WE ${PDLL_FILE} NAME_WE)

  set(PDL_FILE ${CMAKE_BINARY_DIR}/${NAME_WE}.pdl.mlir)
  set(PDL_INTERP_FILE ${CMAKE_BINARY_DIR}/Patterns/${NAME_WE}.pdl_interp.mlir)

  list(APPEND PDL_FILES ${PDL_FILE})
  list(APPEND PDL_INTERP_FILES ${PDL_INTERP_FILE})

  # Step 1: Lower PDLL -> PDL dialect
  add_custom_command(
    OUTPUT ${PDL_FILE}
    COMMAND ${MLIR_BIN_PATH}/mlir-pdll ${PDLL_FILE} -I ${MLIR_MAIN_SRC_DIR}/include -I ${CMAKE_SOURCE_DIR}/include -x=mlir -o ${PDL_FILE}
    DEPENDS ${PDLL_FILE}
    COMMENT "Lowering ${PDLL_FILE} to PDL dialect"
  )

  # Step 2: Lower PDL -> PDLInterp dialect (runtime metadata)
  add_custom_command(
    OUTPUT ${PDL_INTERP_FILE}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Patterns
    COMMAND ${MLIR_BIN_PATH}/mlir-opt ${PDL_FILE} --convert-pdl-to-pdl-interp -o ${PDL_INTERP_FILE}
    DEPENDS ${PDL_FILE}
    COMMENT "Lowering ${PDL_FILE} to PDLInterp dialect"
  )

endforeach()

add_custom_target(GeneratePDLInterp ALL
  DEPENDS ${PDL_INTERP_FILES}
)

# Install pdl_interp.mlir files as runtime metadata.
# FunctionPartitionPass.cpp resolves them at runtime via:
#   getRootPath() + "/Patterns/<name>.pdl_interp.mlir"
# where getRootPath() returns the parent of the directory containing the .so,
# so the files must live at <install-prefix>/lib/Patterns/.
install(FILES ${PDL_INTERP_FILES}
  DESTINATION "Patterns"
)
