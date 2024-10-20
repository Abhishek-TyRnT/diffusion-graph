

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
    let extraClassDefinition = [{
        ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void ReluOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}