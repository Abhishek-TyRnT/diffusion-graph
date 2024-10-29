#ifndef VLLM_GRAPH_TRANSFORM_PASSES_H_
#define VLLM_GRAPH_TRANSFORM_PASSES_H_


include "mlir/Pass/PassBase.td"

def ConvertGlobalFunctionPass : Pass<"convert-global-function-pass", "ModuleOp"> {
    let summary = "This Pass converts global function of Torch IR to vllm_graph supported dtypes.";
    let description = [{This pass is created to make sure that the arguments of the forward function is always vllm_graph_dtypes}];
    
}

#endif //VLLM_GRAPH_TRANSFORM_PASSES_H_