
add_subdirectory(diffusion_graph)

set(LinkedLibs
    MLIRFuncDialect
    MLIRIR
    
   # Dialects
   DiffusionGraphIRDialect
   TorchMLIRTorchDialect
   MLIRSparseTensorDialect

   # Conversion Passes
   DiffusionGraphConversionPasses
   DiffusionGraphTransformPasses

   # MLIR Transforms
   MLIRTransforms

   #Utils
   TorchMLIRTorchUtils
   )

add_mlir_library(DiffusionGraphInitAll
InitAll.cpp

LINK_LIBS PUBLIC
${LinkedLibs}
)
