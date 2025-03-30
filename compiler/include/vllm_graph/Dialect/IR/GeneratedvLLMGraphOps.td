

def vllm_ReluOp : vLLMGraph_Op<"vllm.Relu", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::Relu : (Tensor) -> (Tensor)`";
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
        TensorOrScalarType:$input2,  
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

def vllm_SubOp : vLLMGraph_Op<"vllm.sub", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::sub : (Tensor?, Tensor? , scalar) -> (Tensor)`";
    let arguments = (ins 
        TensorOrScalarType:$input1, 
        TensorOrScalarType:$input2,  
        I32:$alpha);
    
    let results = (outs
                AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult SubOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void SubOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_MulOp : vLLMGraph_Op<"vllm.mul", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::mul : (Tensor, Tensor) -> (Tensor)`";
    let arguments = (ins 
        TensorOrScalarType:$input1, 
        TensorOrScalarType:$input2  
    );
    
    let results = (outs
                TensorOrScalarType:$result);
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void MulOp::print(OpAsmPrinter &printer) {
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
    
    let summary = "Generated op for for `vllm:addmm : (Tensor, Tensor, Tensor, scalar, scalar,) -> Tensor`";
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
    AnyTorchArrayType:$size
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

def vllm_LayerNormOp : vLLMGraph_Op<"vllm.layer_norm", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated Op for `vllm:layer_norm : (Tensor, list, Tensor, Tensor, float, bool) -> Tensor";

    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_ListType:$normliazed_shape,
      TorchTypeOrTensorType:$weight,
      TorchTypeOrTensorType:$bias,
      F32:$epsilon
    );
    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult LayerNormOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void LayerNormOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
    }];
}

def vllm_TanhOp : vLLMGraph_Op<"vllm.tanh", [
        vLLMAllowsTypeRefinement,
        vLLMHasValueSemantics,
        vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::tanh : (Tensor,) -> (Tensor)`";
    let arguments = (ins
        AnyTorchTensorType:$self
    );
    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult TanhOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void TanhOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_PowOp : vLLMGraph_Op<"vllm.pow", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::pow : (Tensor, scalar) -> (Tensor)`";
    let arguments = (ins 
        AnyTorchTensorType:$input,
        ScalarType: $exponent
    );
    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult PowOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void PowOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_EmbeddingOp : vLLMGraph_Op<"vllm.embedding", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
    ]> {
    
    let summary = "Generated op for `vllm::embedding : (Tensor, Tensor) -> Tensor`";
    let arguments = (ins
        AnyTorchTensorType:$input,
        F32Tensor:$weight
        );

    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult EmbeddingOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void EmbeddingOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_BroadCastOp : vLLMGraph_Op<"vllm.broadcast_to", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly]> {

    let summary = "Generated op for `vllm::BroadCastOp : (Tensor, List) -> Tensor`";
    let arguments = (ins
        AnyTorchTensorType:$input,
        Torch_ListType:$shape
        );
    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult BroadCastOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void BroadCastOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];

}

def vllm_PermuteOp : vLLMGraph_Op<"vllm.permute", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
]> {

    let summary = "Generated op for `vllm::permute : (Tensor, list) -> Tensor`";
    let arguments = (ins 
        AnyTorchTensorType:$input,
        Torch_ListType:$shape
    );
    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult PermuteOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void PermuteOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];    
}

def vllm_IndexSelectOp : vLLMGraph_Op<"vllm.index_select", [
    vLLMAllowsTypeRefinement,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::index_select : (Tensor, int, Tesnor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    I32:$dim,
    TorchTypeOrTensorType:$indices
  );
  
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult IndexSelectOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void IndexSelectOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
  }];
}

def vllm_UnsqueezeOp : vLLMGraph_Op<"vllm.unsqueeze", [
    vLLMAllowsTypeRefinement,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::unsqueeze : (Tensor, int) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    I32:$dim
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult UnsqueezeOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void UnsqueezeOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
  }];
}

def vllm_SqueezeOp : vLLMGraph_Op<"vllm.squeeze", [
    vLLMAllowsTypeRefinement,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::squeeze : (Tensor, int) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    I32:$dim
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult SqueezeOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 1, 1);
    }
    void SqueezeOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 1, 1);
    }
  }];
}

def vllm_WhereOp : vLLMGraph_Op<"vllm.where", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for 1 vllm::where : (Tensor, TensororScalar, TensororScalar) -> (Tensor)";
    let arguments = (ins 
        TorchTypeOrTensorType : $input,
        TensorOrScalarType : $trueInput,
        TensorOrScalarType : $falseInput
    );
    
    let results = (outs
        AnyTorchTensorType: $result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult WhereOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 1, 1);
        }
        void WhereOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}