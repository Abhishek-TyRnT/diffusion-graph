

#ifndef DIFFUSION_GRAPH_BASE
#define DIFFUSION_GRAPH_BASE

include "mlir/IR/OpBase.td"

def DiffusionGraphIR_Dialect : Dialect {
    let name = "diffusion_graph";
    let cppNamespace = "::mlir::diffusion_graph";
    let description = [{
        A high level dialect for creating a graph to support for diffusion models.

        This Dialect specifically represents the low level kernels that are specifically called 
        by diffusion model runtimes in an optimised fashion.

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

class DiffusionGraphOpTrait<string name> : NativeOpTrait<""> {
    let trait = name;
    let cppNamespace = "::mlir::diffusion_graph::OpTrait";
}

def diffusionHasValueSemantics : DiffusionGraphOpTrait<"HasValueSemantics">;
def diffusionReadOnly : DiffusionGraphOpTrait<"ReadOnly">;
def diffusionIsTrailingUnderscoreInplaceVariant
    : DiffusionGraphOpTrait<"IsTrailingUnderscoreInplaceVariant">;
def diffusionAllowsTypeRefinement : DiffusionGraphOpTrait<"AllowsTypeRefinement">;
def diffusionAllowedInModuleInitializer : DiffusionGraphOpTrait<"AllowedInModuleInitializer">;

#endif
