set(InterpreterFiles
    PatternInterpreter.cpp
    NativeRewrites.cpp
)

add_mlir_library(
    DiffusionGraphPatternInterpreter
    ${InterpreterFiles}

    LINK_LIBS PUBLIC

    DiffusionGraphIRDialect
    MLIRPDLInterpDialect
    MLIRPDLDialect
    MLIRIR
    MLIRParser
    LLVMSupport

)
