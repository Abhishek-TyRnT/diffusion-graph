set(InterpreterFiles
    PatternInterpreter.cpp
)

add_mlir_library(
    vLLMGraphPatternInterpreter
    ${InterpreterFiles}

    LINK_LIBS PUBLIC

    vLLMGraphIRDialect
    MLIRPDLInterpDialect
    MLIRPDLDialect
    MLIRIR
    MLIRParser
    LLVMSupport

)
