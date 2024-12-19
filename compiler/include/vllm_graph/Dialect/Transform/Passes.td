#ifndef VLLM_GRAPH_TRANSFORM_PASSES_H_
#define VLLM_GRAPH_TRANSFORM_PASSES_H_


include "mlir/Pass/PassBase.td"

def ConvertGlobalFunctionPass : Pass<"convert-global-function-pass", "ModuleOp"> {
    let summary = "This Pass converts global function of Torch IR to vllm_graph supported dtypes.";
    let description = [{This pass is created to make sure that the arguments of the forward function is always vllm_graph_dtypes}];
    
}

def RecomposeSimpleOpsToComplexPass : Pass<"recompose-simple-ops-to-complex", "func::FuncOp"> {
   let summary = "This ops recomposes simple torch-ops to complex vllm-graph operations. ";
   let description = [{
       Torch-MLIR breaks down all layers into its simpler constituent ops. This might not always be 
       optimal as well optimized hardware accelarator kernels often have optimised complex operations.
       Hence a pass is needed to recompose operations into complex operations.

       This pass is important for selecting appropriate kernels during runtime
   }];

   let constructor = "mlir::vllm_graph::createRecomposeSimpleOpsToComplexOps()";

}

#endif //VLLM_GRAPH_TRANSFORM_PASSES_H_