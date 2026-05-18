
#ifndef VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES
#define VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTorchToDiffusionGraph : Pass<"convert-torch-to-diffusion-graph", "func::FuncOp"> {
   let summary = "Converts Torch Ops in the diffusion graph dialect";
   let description = [{
       Converts equivalent torch ops to diffusion-graph Dialect 
   }];

   let constructor = "mlir::diffusion_graph::createTorchToDiffusionGraph()";

}

#endif // VLLM_GRAPH_CONVERISON_TORCH_TO_VLLMGRAPH_PASSES