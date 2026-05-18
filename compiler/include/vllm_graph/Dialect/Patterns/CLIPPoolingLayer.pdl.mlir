

#include "mlir/Dialect/Func/IR/FuncOps.td"
#include "mlir/Dialect/Arith/IR/ArithOps.td"

// Include your custom dialect here
#include "vllm_graph/Dialect/IR/vLLMGraphBase.td"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.td"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.td"

Rewrite CreateCLIPPoolingFunc(ops: Value);

Pattern ExtractPoolingLayer with benefit(1) {

    let ConstantNeg1Op = op<arith.constant>();
    let Constant3Op = op<arith.constant>();
    let ConstantFalseOp = op<arith.constant>();

    let layerNormOp = op<diffusion_graph.torch.layer_norm>;
    let CastDtypeOp = op<diffusion_graph.torch.cast_dtype>(_ : Value, Constant3Op );

    let MaxDimOp = op<diffusion_graph.temp.max_dim>(CastDtypeOp, ConstantNeg1Op , ConstantFalseOp );

    let Indices : Value = MaxDimOp.indices;
    let IndexSelectOp = op<diffusion_graph.torch.index_select>(layerNormOp , _ : Value, Indices);

    rewrite IndexSelectOp with {
      CreateCLIPPoolingFunc(IndexSelectOp);
      erase IndexSelectOp;
      erase MaxDimOp;
      erase CastDtypeOp;
    };

}