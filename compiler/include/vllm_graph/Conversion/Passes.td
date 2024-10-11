
#ifndef VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES
#define VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTorchTovLLMGraph : Pass<"convert-torch-to-vllm-graph", "func::FuncOp"> {
   let summary = "Converts Torch Ops in the vllm graph dialect";
   let description = [{
       Converts equivalent torch ops to vllm-graph Dialect 
   }];

   let constructor = "mlir::vllm_graph::createTorchTovLLMGraph()";

}

#endif // VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES