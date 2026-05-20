// PDLL pattern to extract constant ops and related custom ops into a new FuncOp

#include "mlir/Dialect/Func/IR/FuncOps.td"
#include "mlir/Dialect/Arith/IR/ArithOps.td"

// Include your custom dialect here
#include "diffusion_graph/Dialect/IR/DiffusionGraphBase.td"
#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.td"
#include "diffusion_graph/Dialect/IR/DiffusionGraphTypes.td"



Rewrite CreatePoolingFunc(ops: Value);

// Pattern to identify and extract constant operations along with custom ops
Pattern ExtractPoolingLayer with benefit(1) {
  // Match a function that contains our pattern
  // let root = op<func.func>(
  //   _: SymbolNameAttr,  // function name (don't care)
  //   _: TypeAttr,  // function type
  //   body: region(
  //     // Match operations in the function body
  //     ops: EachOf<[
        // Match constant operations
  let constantOp1 = op<arith.constant>() ;

  let constantOp0 = op<arith.constant>() ;

  let constantZeroTensor = op<diffusion_graph.constant.tensor>(); 

  let constantBias = op<diffusion_graph.constant.tensor>();

  let constantWeight = op<diffusion_graph.constant.tensor>();
    
    // Match custom dialect operations that use constants
    // Replace "custom.op" with your actual custom op
  let layerNormOp = op<diffusion_graph.torch.layer_norm>;

  let IndexSelectOp = op<diffusion_graph.torch.index_select>(layerNormOp, constantOp1, constantZeroTensor);

  let SqueezeOp = op<diffusion_graph.torch.squeeze>(IndexSelectOp, constantOp1) ;

  let TransposeOp = op<diffusion_graph.torch.transpose>(constantBias, constantOp0, constantOp1) ;

  let AddMmOp = op<diffusion_graph.torch.addmm>(constantWeight, SqueezeOp, TransposeOp, constantOp1, constantOp1) ;

  let TanhOp = op<diffusion_graph.torch.tanh>(AddMmOp);
                      
  rewrite IndexSelectOp with {
    CreatePoolingFunc(IndexSelectOp);
    erase TanhOp;
    erase AddMmOp;
    erase TransposeOp;
    erase SqueezeOp;
    erase IndexSelectOp;
  };

}