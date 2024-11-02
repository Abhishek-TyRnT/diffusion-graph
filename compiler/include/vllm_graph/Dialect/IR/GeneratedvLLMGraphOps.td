

def vllm_ReluOp : vLLMGraph_Op<"vllm.Relu", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
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


def vllm_AddOp : vLLMGraph_Op<"vllm.add", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::Add : (Tensor, Tensor , scalar) -> (Tensor)`";
    let arguments = (ins 
        AnyTorchTensorType:$input1, 
        AnyTorchTensorType:$input2,  
        I32:$alpha);
    
    let results = (outs
                AnyTorchTensorType:$result);
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void AddOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];


}

def vllm_TransposeOp : vLLMGraph_Op<"vllm.transpose", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::Transpose : (Tensor, scalar, scalar) -> (Tensor)`";
    let arguments = (ins
        AnyTorchTensorType:$input,
        I32:$dim0,
        I32:$dim1
        );
    
    let results = (outs 
            AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult TransposeOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void TransposeOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
    }