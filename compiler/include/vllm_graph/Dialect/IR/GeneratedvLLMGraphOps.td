

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
        TorchTypeOrTensorType:$input2,  
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
        TorchTypeOrTensorType:$input,
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

def vllm_MatmulOp : vLLMGraph_Op<"vllm.matmul", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm:matmul : (Tensor, Tensor) -> (Tensor)`:";
    let arguments = (ins
        AnyTorchTensorType:$input,
        TorchTypeOrTensorType:$weight);
    
    let results = (outs
        AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void MatmulOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
}

def vllm_BMMOp : vLLMGraph_Op<"vllm.bmm", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm:matmul : (Tensor, Tensor) -> (Tensor)`:";
    let arguments = (ins
        AnyTorchTensorType:$input,
        TorchTypeOrTensorType:$weight);
    
    let results = (outs
        AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult BMMOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void BMMOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
}

def vllm_SoftmaxOp : vLLMGraph_Op<"vllm.softmax", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {

    let summary = "Generated op for `vllm:softmax : (Tensor, int) -> Tensor";
    let arguments = (ins
        AnyTorchTensorType:$input,
        I32:$dim
        );

    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult SoftmaxOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void SoftmaxOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
}

def vllm_AddmmOp : vLLMGraph_Op<"vllm.addmm", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for for `vllm:addmm : (Tensor, Tensor, Tensor, scalar, scalar,) -> Tensor";
    let arguments = (ins
        TorchTypeOrTensorType:$bias,
        AnyTorchTensorType:$input,
        TorchTypeOrTensorType:$weight,
        ScalarType:$alpha,
        ScalarType:$beta
        );
    
    let results = (outs
        AnyTorchTensorType:$result
        );
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult AddmmOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void AddmmOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_ViewOp : vLLMGraph_Op<"vllm.view", [
    vLLMAllowsTypeRefinement,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::view : (Tensor, Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    Torch_TupleType:$size
  );
  let results = (outs
    AnyTorchOptionalTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult ViewOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 2, 1);
    }
    void ViewOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 2, 1);
    }
  }];
}

def vllm_ConstTupleOp : vLLMGraph_Op<"vllm.const_tuple",[
    vLLMAllowsTypeRefinement,
    vLLMReadOnly
  ]> {

    let summary = "Generated Op for `vllm:const_tuple : () -> TorchTuple";
    let arguments = (ins
        ElementsAttr:$value    // A required attribute to hold the constant value.
    );
    let results = (outs
        Torch_TupleType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let builders = [
        // Build a constant with a given constant tensor value.
        OpBuilder<(ins "DenseElementsAttr":$value), [{
        build($_builder, $_state, value.getType(), value);
        }]>
    ];
    let extraClassDefinition = [{
        ParseResult ConstTupleOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
        }
        void ConstTupleOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
        }
  }];
}



def vllm_DivScalarOp : vLLMGraph_Op<"vllm.div.scalar", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {

    let summary = "Generated op for `vllm:div.scalar : (Tensor, int) -> Tensor";
    let arguments = (ins
        TorchTypeOrTensorType:$input,
        F32:$scalar
        );

    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult DivScalarOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void DivScalarOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
}