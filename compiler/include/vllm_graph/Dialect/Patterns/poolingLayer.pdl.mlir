// PDLL pattern to extract constant ops and related custom ops into a new FuncOp

#include "mlir/Dialect/Func/IR/FuncOps.td"
#include "mlir/Dialect/Arith/IR/ArithOps.td"

// Include your custom dialect here
#include "vllm_graph/Dialect/IR/vLLMGraphBase.td"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.td"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.td"



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

  let constantZeroTensor = op<vllm_graph.constant.tensor>(); 

  let constantBias = op<vllm_graph.constant.tensor>();

  let constantWeight = op<vllm_graph.constant.tensor>();
    
    // Match custom dialect operations that use constants
    // Replace "custom.op" with your actual custom op
  let layerNormOp = op<vllm_graph.vllm.layer_norm>;

  let IndexSelectOp = op<vllm_graph.vllm.index_select>(layerNormOp, constantOp1, constantZeroTensor);

  let SqueezeOp = op<vllm_graph.vllm.squeeze>(IndexSelectOp, constantOp1) ;

  let TransposeOp = op<vllm_graph.vllm.transpose>(constantBias, constantOp0, constantOp1) ;

  let AddMmOp = op<vllm_graph.vllm.addmm>(constantWeight, SqueezeOp, TransposeOp, constantOp1, constantOp1) ;

  let TanhOp = op<vllm_graph.vllm.tanh>(AddMmOp);
                      
  rewrite IndexSelectOp with {
    CreatePoolingFunc(IndexSelectOp);
    erase TanhOp;
    erase AddMmOp;
    erase TransposeOp;
    erase SqueezeOp;
    erase IndexSelectOp;
    
    
    
    

  };

}