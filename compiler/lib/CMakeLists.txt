
add_subdirectory(vllm_graph)

set(LinkedLibs
    MLIRFuncDialect
    MLIRIR
    
   # Dialects
   vLLMGraphIRDialect
   TorchMLIRTorchDialect
   MLIRSparseTensorDialect

   # Conversion Passes
   vLLMGraphConversionPasses
   vLLMGraphTransformPasses

   #Utils
   TorchMLIRTorchUtils
   )

add_mlir_library(vLLMGraphInitAll
InitAll.cpp

LINK_LIBS PUBLIC
${LinkedLibs}
)
