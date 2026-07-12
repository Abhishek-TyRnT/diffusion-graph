

def diffusion_ReluOp : DiffusionGraph_Op<"torch.Relu", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::Relu : (Tensor) -> (Tensor)`";
    let arguments = (ins
        AnyTorchTensorType:$self
    );
    let results = (outs
        AnyTorchOptionalTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void ReluOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}


def diffusion_AddOp : DiffusionGraph_Op<"torch.add", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
    
    let summary = "Generated op for `torch::Add : (Tensor, Tensor , scalar) -> (Tensor)`";
    let arguments = (ins 
        AnyTorchTensorType:$input1, 
        TensorOrScalarType:$input2,  
        I32:$alpha);
    
    let results = (outs
                AnyTorchTensorType:$result);
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void AddOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_SubOp : DiffusionGraph_Op<"torch.sub", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::sub : (Tensor?, Tensor? , scalar) -> (Tensor)`";
    let arguments = (ins 
        TensorOrScalarType:$input1, 
        TensorOrScalarType:$input2,  
        ScalarType:$alpha);
    
    let results = (outs
                AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult SubOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void SubOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_MulOp : DiffusionGraph_Op<"torch.mul", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::mul : (Tensor, Tensor) -> (Tensor)`";
    let arguments = (ins 
        TensorOrScalarType:$input1, 
        TensorOrScalarType:$input2  
    );
    
    let results = (outs
                TensorOrScalarType:$result);
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void MulOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];

    let hasFolder = 1;
}

def diffusion_TransposeOp : DiffusionGraph_Op<"torch.transpose", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::Transpose : (Tensor, scalar, scalar) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void TransposeOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
    }];
}

def diffusion_MatmulOp : DiffusionGraph_Op<"torch.matmul", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `diffusion:matmul : (Tensor, Tensor) -> (Tensor)`:";
    let arguments = (ins
        AnyTorchTensorType:$input,
        TorchTypeOrTensorType:$weight);
    
    let results = (outs
        AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void MatmulOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
    }];
}

def diffusion_BMMOp : DiffusionGraph_Op<"torch.bmm", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `diffusion:matmul : (Tensor, Tensor) -> (Tensor)`:";
    let arguments = (ins
        AnyTorchTensorType:$input,
        TorchTypeOrTensorType:$weight);
    
    let results = (outs
        AnyTorchTensorType:$result);
    
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult BMMOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void BMMOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
    }];
}

def diffusion_SoftmaxOp : DiffusionGraph_Op<"torch.softmax", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {

    let summary = "Generated op for `diffusion:softmax : (Tensor, int) -> Tensor";
    let arguments = (ins
        AnyTorchTensorType:$input,
        I32:$dim
        );

    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult SoftmaxOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void SoftmaxOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
    }];
}

def diffusion_AddmmOp : DiffusionGraph_Op<"torch.addmm", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for for `diffusion:addmm : (Tensor, Tensor, Tensor, scalar, scalar,) -> Tensor`";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void AddmmOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_ViewOp : DiffusionGraph_Op<"torch.view", [
    diffusionAllowsTypeRefinement,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::view : (Tensor, Tensor) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
    }
    void ViewOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 2, 1);
    }
  }];
}

def diffusion_ConstTupleOp : DiffusionGraph_Op<"torch.const_tuple",[
    diffusionAllowsTypeRefinement,
    diffusionReadOnly
  ]> {

    let summary = "Generated Op for `diffusion:const_tuple : () -> TorchTuple";
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
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
        }
        void ConstTupleOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
        }
  }];
}



def diffusion_DivScalarOp : DiffusionGraph_Op<"torch.div.scalar", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {

    let summary = "Generated op for `diffusion:div.scalar : (Tensor, int) -> Tensor";
    let arguments = (ins
        TorchTypeOrTensorType:$input,
        ScalarType:$scalar
        );

    let results = (outs
        AnyTorchTensorType:$result);

    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
    ParseResult DivScalarOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
    }
    void DivScalarOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
    }
    }];
}

def diffusion_LayerNormOp : DiffusionGraph_Op<"torch.layer_norm", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated Op for `diffusion:layer_norm : (Tensor, list, Tensor, Tensor, float, bool) -> Tensor";

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
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void LayerNormOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
    }];
}

def diffusion_TanhOp : DiffusionGraph_Op<"torch.tanh", [
        diffusionAllowsTypeRefinement,
        diffusionHasValueSemantics,
        diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::tanh : (Tensor,) -> (Tensor)`";
    let arguments = (ins
        AnyTorchTensorType:$self
    );
    let results = (outs
        AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
        ParseResult TanhOp::parse(OpAsmParser &parser, OperationState &result) {
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void TanhOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_PowOp : DiffusionGraph_Op<"torch.pow", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::pow : (Tensor, scalar) -> (Tensor)`";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void PowOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_EmbeddingOp : DiffusionGraph_Op<"torch.embedding", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
    ]> {
    
    let summary = "Generated op for `torch::embedding : (Tensor, Tensor) -> Tensor`";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void EmbeddingOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_BroadCastOp : DiffusionGraph_Op<"torch.broadcast_to", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly]> {

    let summary = "Generated op for `torch::BroadCastOp : (Tensor, List) -> Tensor`";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void BroadCastOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];

}

def diffusion_PermuteOp : DiffusionGraph_Op<"torch.permute", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
]> {

    let summary = "Generated op for `torch::permute : (Tensor, list) -> Tensor`";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void PermuteOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];    
}

def diffusion_IndexSelectOp : DiffusionGraph_Op<"torch.index_select", [
    diffusionAllowsTypeRefinement,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::index_select : (Tensor, int, Tesnor) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void IndexSelectOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
  }];
}

def diffusion_UnsqueezeOp : DiffusionGraph_Op<"torch.unsqueeze", [
    diffusionAllowsTypeRefinement,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::unsqueeze : (Tensor, int) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void UnsqueezeOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
  }];
}

def diffusion_SqueezeOp : DiffusionGraph_Op<"torch.squeeze", [
    diffusionAllowsTypeRefinement,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::squeeze : (Tensor, int) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
    }
    void SqueezeOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 1, 1);
    }
  }];
}

def diffusion_WhereOp : DiffusionGraph_Op<"torch.where", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for torch::where : (Tensor, TensororScalar, TensororScalar) -> (Tensor)";
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
            return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
        }
        void WhereOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_DtypeCastOp : DiffusionGraph_Op<"torch.cast_dtype",[
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
]> {

    let summary = "Generated op for torch::cast_dtype : (Tensor, int ) -> (Tensor)";
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
            return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
        }
        void DtypeCastOp::print(OpAsmPrinter &printer) {
            printDefaultDiffusionGraphOp(printer, *this, 1, 1);
        }
    }];
}

def diffusion_ScaledDotProductAttentionOp : DiffusionGraph_Op<"torch.scaled_dot_product_attention", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::scaled_dot_product_attention : (Tensor, Tensor, Tensor, Tensor?, float, bool, float?, bool) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 8, 1);
    }
    void ScaledDotProductAttentionOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 8, 1);
    }
  }];
}

def diffusion_OnesOp : DiffusionGraph_Op<"torch.ones", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::ones : (int[], int?, int?) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 5, 1);
    }
    void OnesOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 5, 1);
    }
  }];
}

def diffusion_SizeOp : DiffusionGraph_Op<"torch.size", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
  let summary = "Generated op for `torch::size : (Tensor, int) -> (int)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
    }
    void SizeOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 2, 1);
    }
  }];

  let hasFolder = 1;
}

def diffusion_ArangeOp : DiffusionGraph_Op<"temp.arange", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
  
  let summary = "Generated op for `torch::arange : (int, int, int, int) -> (Tensor)`";
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
      return parseDefaultDiffusionGraphOp(parser, result, 4, 1);
    }
    void ArangeOp::print(OpAsmPrinter &printer) {
      printDefaultDiffusionGraphOp(printer, *this, 4, 1);
    }
  }];
}

def diffusion_Conv2DOp : DiffusionGraph_Op<"torch.conv2d", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::conv2d : (Tensor, Tensor, Tensor, int[], int[]) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 5, 1);
      }
      void Conv2DOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 5, 1);
      }
    }];
}

def diffusion_GroupNormOp : DiffusionGraph_Op<"torch.group_norm", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::group_norm : (Tensor, int, Tensor, Tensor, float) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 5, 1);
      }
      void GroupNormOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 5, 1);
      }
    }];
}

def diffusion_SiLUOp : DiffusionGraph_Op<"torch.silu", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::silu : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SiLUOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void SiLUOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_SplitOp : DiffusionGraph_Op<"torch.split", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::split : (Tensor, int, int) -> (List<Tensor>)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 3, 1);
      }
      void SplitOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 3, 1);
      }
    }];
}

def diffusion_GeluOp : DiffusionGraph_Op<"torch.gelu", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::gelu : (Tensor, str) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
      }
      void GeluOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
      }
    }];
}

def diffusion_ExpOp : DiffusionGraph_Op<"torch.exp", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::exp : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult ExpOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void ExpOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_SinOp : DiffusionGraph_Op<"torch.sin", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::sin : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SinOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void SinOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_CosOp : DiffusionGraph_Op<"torch.cos", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::cos : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult CosOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void CosOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_ConcatenateOp : DiffusionGraph_Op<"torch.concatenate", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::concatenate : (List<Tensor>, int) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
      }
      void ConcatenateOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
      }
    }];
}

def diffusion_SumOp : DiffusionGraph_Op<"torch.sum", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::sum : (Tensor, list[int], bool) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 3, 1);
      }
      void SumOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 3, 1);
      }
    }];
}

def diffusion_RSqrtOp : DiffusionGraph_Op<"torch.rsqrt", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::rsqrt : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult RSqrtOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void RSqrtOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_BroadCastIndexOp : DiffusionGraph_Op<"temp.broadcast_index", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::broadcast_index : (Tensor, list[Tensor]) -> (Tensor)`. This is a temporary op for hacked_twin_index.";
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
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
      }
      void BroadCastIndexOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
      }
    }];
}

def diffusion_UpsampleOp : DiffusionGraph_Op<"torch.upsample", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::upsample : (Tensor, scalar, string) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      ScalarType:$scale_factor,
      Torch_StringType:$interpolateStyle
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult UpsampleOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 3, 1);
      }
      void UpsampleOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 3, 1);
      }
    }];
}

def diffusion_SigmoidOp : DiffusionGraph_Op<"torch.sigmoid", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::sigmoid : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult SigmoidOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void SigmoidOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_LessThanOp : DiffusionGraph_Op<"torch.less_than", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::less_than : (Tensor, Tensor) -> (Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 2, 1);
      }
      void LessThanOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 2, 1);
      }
    }];
}

def diffusion_ContiguousOp : DiffusionGraph_Op<"torch.contiguous", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::contiguous : (Tensor) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult ContiguousOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 1, 1);
      }
      void ContiguousOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 1, 1);
      }
    }];
}

def diffusion_MaxDimOp : DiffusionGraph_Op<"temp.max_dim", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::max_dim : (Tensor, int, bool) -> (Tensor, Tensor)`";
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
        return parseDefaultDiffusionGraphOp(parser, result, 3, 2);
      }
      void MaxDimOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 3, 2);
      }
    }];
}

def diffusion_NarrowOp : DiffusionGraph_Op<"torch.narrow", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::narrow : (Tensor, int, int, int) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      I32:$dim,
      I32:$start,
      I32:$length
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult NarrowOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 4, 1);
      }
      void NarrowOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 4, 1);
      }
    }];
}

def diffusion_PadOp : DiffusionGraph_Op<"torch.pad", [
    diffusionAllowsTypeRefinement,
    diffusionHasValueSemantics,
    diffusionReadOnly
  ]> {
    let summary = "Generated op for `torch::pad : (Tensor, tuple, string, float) -> (Tensor)`";
    let arguments = (ins
      AnyTorchTensorType:$input,
      Torch_ListType:$pad,
      Torch_StringType:$mode,
      F32:$value
    );
    let results = (outs
      AnyTorchTensorType:$result
    );
    let hasCustomAssemblyFormat = 1;
    let extraClassDefinition = [{
      ParseResult PadOp::parse(OpAsmParser &parser, OperationState &result) {
        return parseDefaultDiffusionGraphOp(parser, result, 4, 1);
      }
      void PadOp::print(OpAsmPrinter &printer) {
        printDefaultDiffusionGraphOp(printer, *this, 4, 1);
      }
    }];
}
    
    

