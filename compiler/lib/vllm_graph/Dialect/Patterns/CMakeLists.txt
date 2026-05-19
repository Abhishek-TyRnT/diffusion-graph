set(InterpreterFiles
    PatternInterpreter.cpp
    NativeRewrites.cpp
)

add_mlir_library(
    vLLMGraphPatternInterpreter
    ${InterpreterFiles}

    LINK_LIBS PUBLIC

    DiffusionGraphIRDialect
    MLIRPDLInterpDialect
    MLIRPDLDialect
    MLIRIR
    MLIRParser
    LLVMSupport

)
