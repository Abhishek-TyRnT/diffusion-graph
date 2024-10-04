

def vllm_ReluOp : vLLMGraph_Op<"vllm.Relu", [
        AllowsTypeRefinement,
        HasValueSemantics,
        ReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::Relu : (Tensor, Scalar, Scalar) -> (Tensor)`";
    let arguments = (ins
        AnyTorchTensorType:$self
    );
    let results = (outs
        AnyTorchOptionalTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
}