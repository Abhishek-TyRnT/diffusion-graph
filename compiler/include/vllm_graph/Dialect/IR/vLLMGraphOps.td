
#ifndef VLLMGRAPH_OPS
#define VLLMGRAPH_OPS

include "vllm_graph/Dialect/IR/vLLMGraphBase.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "vllm_graph/Dialect/IR/vLLMGraphTypes.td"

class vLLMGraph_Op<string mnenomic, list<Trait> traits = []>
    : Op<vLLMGraphIR_Dialect, mnenomic, traits> {
}

include "vllm_graph/Dialect/IR/GeneratedvLLMGraphOps.td"

def vllm_ConstantDeviceOp : vLLMGraph_Op<"constant.device",
    [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Materialize a constant Device value.";
  let description = [{
    Info regarding device which a given op should run on
  }];
  let arguments = (ins
    StrAttr:$value
  );
  let results = (outs
    Torch_DeviceType:$result
  );
  let assemblyFormat = "$value attr-dict";
}

def vllm_CastOp : vLLMGraph_Op<"vllm.cast_op",[Pure]> {
  let summary = "Acts as temporary cast Op when Dialect is being transformed";
  let description = [{
    When Torch Dialect is being lowered to vllm_graph Dialect, For chaning just function signatures,
    This Op acts a buffer between them, but will be removed in downstream passes. 
  }];
    let arguments = (ins
        AnyType:$input
    );
    let results = (outs
        AnyType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult CastOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void CastOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}


def vllm_ListOp: vLLMGraph_Op<"vllm.list_op", [
    Pure,
    vLLMAllowsTypeRefinement
  ]> {
  let summary = "vllm representation of a list";

  let arguments = (ins
    Variadic<AnyTorchType>:$elements
  );
  let results = (outs
    Torch_ListType:$result
  );

  let assemblyFormat = [{
    $elements attr-dict `:` functional-type(operands, results)
  }];
}


#endif // VLLMGRAPH_OPS
