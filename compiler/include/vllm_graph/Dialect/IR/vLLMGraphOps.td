
#ifndef VLLMGRAPH_OPS
#define VLLMGRAPH_OPS

include "vllm_graph/Dialect/IR/vLLMGraphBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

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

#endif // VLLMGRAPH_OPS
