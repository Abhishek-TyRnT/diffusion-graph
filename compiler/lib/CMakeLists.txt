
add_subdirectory(vllm_graph)

set(LinkedLibs
    MLIRFuncDialect
    MLIRIR
    
   # Dialects
   vLLMGraphIRDialect

   # Conversion Passes
   vLLMGraphTorchTovLLMGraph
   )

add_mlir_library(vLLMGraphInitAll
InitAll.cpp

LINK_LIBS PUBLIC
${LinkedLibs}
)
