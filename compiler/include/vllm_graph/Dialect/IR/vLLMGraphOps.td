
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
include "mlir/IR/OpBase.td"


class vLLMGraph_Op<string mnenomic, list<Trait> traits = []>
    : Op<vLLMGraphIR_Dialect, mnenomic, traits> {
}

include "vllm_graph/Dialect/IR/GeneratedvLLMGraphOps.td"

def vllm_ConstantDeviceOp : vLLMGraph_Op<"constant.device",
    [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    ConstantLike]> {
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
  let hasFolder = 1;
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

def vllm_ValueTensorLiteralOp : vLLMGraph_Op<"constant.tensor", [
    ConstantLike,
    Pure,
  ]> {
  let summary = "Create a value of !vllm_graph.constant.tensor type from a literal";
  let description = [{
    Example:
    ```
    Constant tensor storage for parameters
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Torch_ValueTensorType:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
}

def vllm_ConstantNoneOp : vLLMGraph_Op<"constant.none", [
    ConstantLike,
    Pure,
  ]> {
  let summary = "Get the singleton None value.";
  let description = [{
    Not to be confused with the `mlir::NoneType`. Be careful to use
    `vllm::NoneType` to avoid namespace ambiguity.
  }];
  let arguments = (ins);
  let results = (outs Torch_NoneType:$result);
  let assemblyFormat = "attr-dict";
  let hasFolder = 1;
}



#endif // VLLMGRAPH_OPS
