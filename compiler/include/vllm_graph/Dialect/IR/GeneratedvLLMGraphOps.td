

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
        ScalarType:$alpha);
    
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

    let hasFolder = 1;
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

  let hasCanonicalizer = 1;
  
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
        ScalarType:$scalar
        );

    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult DivScalarOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
    }
    void DivScalarOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
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
        AnyTorchTensorType:$weight
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

    let hasCanonicalizer = 1;
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
    let summary = "Generated op for vllm::where : (Tensor, TensororScalar, TensororScalar) -> (Tensor)";
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

def vllm_DtypeCastOp : vLLMGraph_Op<"vllm.cast_dtype",[
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
]> {

    let summary = "Generated op for vllm::cast_dtype : (Tensor, int ) -> (Tensor)";
    let arguments = (ins
        AnyTorchTensorType: $input,
        I32: $dtype
    );

    let results = (outs
        AnyTorchTensorType: $result
    );

    let hasCanonicalizer = 1;

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult DtypeCastOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultvLLMGraphOp(parser, result, 2, 1);
        }
        void DtypeCastOp::print(OpAsmPrinter &printer) {
            printDefaultvLLMGraphOp(printer, *this, 1, 1);
        }
    }];
}

def vllm_ScaledDotProductAttentionOp : vLLMGraph_Op<"vllm.scaled_dot_product_attention", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?, bool) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$query,
    AnyTorchTensorType:$key,
    AnyTorchTensorType:$value,
    AnyTorchOptionalTensorType:$attn_mask,
    F32:$dropout_p,
    I1:$is_causal,
    OptionalF32Type:$scale,
    I1:$enable_gqa
  );
  let results = (outs
    AnyTorchOptionalTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult ScaledDotProductAttentionOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 8, 1);
    }
    void ScaledDotProductAttentionOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 8, 1);
    }
  }];
}

def vllm_OnesOp : vLLMGraph_Op<"vllm.ones", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::ones : (int[], int?, int?) -> (Tensor)`";
  let arguments = (ins
    AnyTorchArrayType:$size,
    OptionalIntType:$dtype,
    OptionalIntType:$layout
  );
  let results = (outs
    AnyTorchOptionalTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult OnesOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 5, 1);
    }
    void OnesOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 5, 1);
    }
  }];
}

def vllm_SizeOp : vLLMGraph_Op<"vllm.size", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
  let summary = "Generated op for `vllm::size : (Tensor, int) -> (int)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    I32:$dim
  );
  let results = (outs
    I32:$result
  );

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult SizeOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 2, 1);
    }
    void SizeOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 2, 1);
    }
  }];

  let hasFolder = 1;
}

def vllm_ArangeOp : vLLMGraph_Op<"temp.arange", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
  
  let summary = "Generated op for `vllm::arange : (int, int, int, int) -> (Tensor)`";
  let arguments = (ins
    I32:$start_index,
    I32:$end_index,
    I32:$step,
    I32:$dtype
  );

  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult ArangeOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultvLLMGraphOp(parser, result, 4, 1);
    }
    void ArangeOp::print(OpAsmPrinter &printer) {
      printDefaultvLLMGraphOp(printer, *this, 4, 1);
    }
  }];
}

def vllm_Conv2DOp : vLLMGraph_Op<"vllm.conv2d", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::conv2d : (Tensor, Tensor, Tensor, int[], int[]) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      AnyTorchTensorType:$weight,
      AnyTorchTensorType:$bias,
      Torch_ListType:$stride,
      Torch_ListType:$padding
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult Conv2DOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 5, 1);
      }
      void Conv2DOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 5, 1);
      }
    }];
}

def vllm_GroupNormOp : vLLMGraph_Op<"vllm.group_norm", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::group_norm : (Tensor, int, Tensor, Tensor, float) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      I32:$num_groups,
      AnyTorchTensorType:$weight,
      AnyTorchTensorType:$bias,
      F32:$epsilon    
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult GroupNormOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 5, 1);
      }
      void GroupNormOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 5, 1);
      }
    }];
}

def vllm_SiLUOp : vLLMGraph_Op<"vllm.silu", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::silu : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SiLUOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void SiLUOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_SplitOp : vLLMGraph_Op<"vllm.split", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::split : (Tensor, int, int) -> (List<Tensor>)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      I32:$split_size_or_sections,
      I32:$dim
    );
    let results = (outs
      Torch_ListType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SplitOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 3, 1);
      }
      void SplitOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 3, 1);
      }
    }];
}

def vllm_GeluOp : vLLMGraph_Op<"vllm.gelu", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::gelu : (Tensor, str) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_StringType:$approximation_mode
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult GeluOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
      }
      void GeluOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
      }
    }];
}

def vllm_ExpOp : vLLMGraph_Op<"vllm.exp", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::exp : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult ExpOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void ExpOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_SinOp : vLLMGraph_Op<"vllm.sin", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::sin : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SinOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void SinOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_CosOp : vLLMGraph_Op<"vllm.cos", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::cos : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult CosOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void CosOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_ConcatenateOp : vLLMGraph_Op<"vllm.concatenate", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::concatenate : (List<Tensor>, int) -> (Tensor)`";
    let arguments = (ins
      Torch_ListType:$inputs,
      I32:$dim
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult ConcatenateOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
      }
      void ConcatenateOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
      }
    }];
}

def vllm_SumOp : vLLMGraph_Op<"vllm.sum", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::sum : (Tensor, list[int], bool) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_ListType:$dim,
      I1:$keepdim
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SumOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 3, 1);
      }
      void SumOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 3, 1);
      }
    }];
}

def vllm_RSqrtOp : vLLMGraph_Op<"vllm.rsqrt", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::rsqrt : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult RSqrtOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void RSqrtOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_BroadCastIndexOp : vLLMGraph_Op<"temp.broadcast_index", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::broadcast_index : (Tensor, list[Tensor]) -> (Tensor)`. This is a temporary op for hacked_twin_index.";
    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_ListType:$indices
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult BroadCastIndexOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
      }
      void BroadCastIndexOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
      }
    }];
}

def vllm_UpsampleOp : vLLMGraph_Op<"vllm.upsample", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::upsample : (Tensor, tuple[int], string) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_TupleType:$size,
      Torch_StringType:$interpolateStyle
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult UpsampleOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 3, 1);
      }
      void UpsampleOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 3, 1);
      }
    }];
}

def vllm_SigmoidOp : vLLMGraph_Op<"vllm.sigmoid", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::sigmoid : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SigmoidOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 1, 1);
      }
      void SigmoidOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 1, 1);
      }
    }];
}

def vllm_LessThanOp : vLLMGraph_Op<"vllm.less_than", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::less_than : (Tensor, Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      AnyTorchTensorType:$other
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult LessThanOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 2, 1);
      }
      void LessThanOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 2, 1);
      }
    }];
}

def vllm_MaxDimOp : vLLMGraph_Op<"temp.max_dim", [
    vLLMAllowsTypeRefinement,
    vLLMHasValueSemantics,
    vLLMReadOnly
  ]> {
    let summary = "Generated op for `vllm::max_dim : (Tensor, int, bool) -> (Tensor, Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      I32:$dim,
      I1:$keepdim
    );
    let results = (outs
      AnyTorchTensorType:$result,
      AnyTorchTensorType:$indices
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult MaxDimOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultvLLMGraphOp(parser, result, 3, 2);
      }
      void MaxDimOp::print(OpAsmPrinter &printer) {
        printDefaultvLLMGraphOp(printer, *this, 3, 2);
      }
    }];
}
