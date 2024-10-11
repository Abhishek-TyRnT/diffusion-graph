

#ifndef VLLM_GRAPH_BASE
#define VLLM_GRAPH_BASE

include "mlir/IR/OpBase.td"

def vLLMGraphIR_Dialect : Dialect {
    let name = "vllm_graph";
    let cppNamespace = "::mlir::vllm_graph";
    let description = [{
        A high level dialect for creating a graph to support for vllm runtime

        This Dialect specifically represents the low level kernels that are specifically called 
        by vllm in an optimised fashion.

        - The tensor sizes have been analysed and static ranks inferred where possile and propagated thoughout the program.

    }];

    //let hasRegionArgAttrVerify = 1;
    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 0;

    let extraClassDeclaration = [{
        /// Parse a type registered to this dialect.
        Type parseType(DialectAsmParser &parser) const override;
        /// Print a type registered to this dialect.
        void printType(Type type, DialectAsmPrinter &printer) const override;
    }];
}

class vLLMGraphOpTrait<string name> : NativeOpTrait<""> {
    let trait = name;
    let cppNamespace = "::mlir::vllm_graph::OpTrait";
}

def vLLMHasValueSemantics : vLLMGraphOpTrait<"HasValueSemantics">;
def vLLMReadOnly : vLLMGraphOpTrait<"ReadOnly">;
def vLLMIsTrailingUnderscoreInplaceVariant
    : vLLMGraphOpTrait<"IsTrailingUnderscoreInplaceVariant">;
def vLLMAllowsTypeRefinement : vLLMGraphOpTrait<"AllowsTypeRefinement">;
def vLLMAllowedInModuleInitializer : vLLMGraphOpTrait<"AllowedInModuleInitializer">;

#endif
