
add_subdirectory(Conversion)
add_subdirectory(Dialect)

set(LinkedLibs
    MLIRFuncDialect
    MLIRIR
    
   # Dialects
   vLLMGraphIRDialect

   # Conversion Passes
   vLLMGraphTorchTovLLMGraph
   )

add_mlir_library(TorchMLIRInitAll
InitAll.cpp

LINK_LIBS PUBLIC
${LinkedLibs}
)
