
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "../PassDetail.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Utils/Utils.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include <vector>
#include <numeric>
#include <optional>
#include <cassert>

using namespace mlir;
using namespace llvm;
using namespace mlir::vllm_graph;
using namespace mlir::torch::Torch;


namespace {

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename OpT>
class EraseOp : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;
  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenReluOp>::matchAndRewrite(
    mlir::torch::Torch::AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value self = adaptor.getSelf();
    MLIRContext *context = op.getContext();
    auto selfTy = cast<TensorType>(self.getType());
    if (!selfTy) {
        return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in vllm_graph");
    }

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    self.setType(convertor->convertType(self.getType()));
    // result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::ReluOp>(op, resultType, self);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenTanhOp>::matchAndRewrite(
    mlir::torch::Torch::AtenTanhOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value self = adaptor.getSelf();
    MLIRContext *context = op.getContext();
    auto selfTy = cast<TensorType>(self.getType());
    if (!selfTy) {
        return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in vllm_graph");
    }

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    self.setType(convertor->convertType(self.getType()));
    // result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::TanhOp>(op, resultType, self);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantDeviceOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantDeviceOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    llvm::StringRef device = op.getValue();

    MLIRContext *context = getContext();
    auto device_type = vllm_graph::DeviceType::get(context);
    rewriter.replaceOpWithNewOp<vllm_graph::ConstantDeviceOp>(op, device_type, device);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    int64_t constant = op.getValue();
    Type intType = rewriter.getIntegerType(32);
    Value result = op.getResult();
    // result.setType(intType);
    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, constant, intType);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantBoolOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    bool constant = op.getValue();
    Type boolType = rewriter.getI1Type();

    Value result = op.getResult();

    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, boolType, rewriter.getBoolAttr(constant));
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantNoneOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantNoneOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    // auto attr = rewriter.getUnitAttr();
    // Type Nonetype = rewriter.getNoneType();

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    
    Value arithOp = rewriter.replaceOpWithNewOp<vllm_graph::ConstantNoneOp>(op, resultType);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantFloatOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    llvm::APFloat constant = op.getValue();
    const llvm::fltSemantics &semantics = constant.getSemantics();
    bool losesInfo = true;
    if (&semantics == &llvm::APFloat::IEEEdouble()) {
        auto status = constant.convert(
            llvm::APFloat::IEEEsingle(), // Target type: single-precision
            llvm::APFloat::rmNearestTiesToEven, // Rounding mode
            &losesInfo
        );

        // Check if the conversion was successful
        if (status != llvm::APFloat::opOK) 
            llvm::errs() << "Warning: Precision loss or rounding occurred during conversion.\n";
    
    }
    
    mlir::FloatType floatType = rewriter.getF32Type();
    Value result = op.getResult();
    // result.setType(floatType);

    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(op, constant, floatType);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenDivScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenDivScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

            
    Value input = op.getOperand(0);
    Value scalar = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    input.setType(convertor->convertType(input.getType()));
    scalar.setType(convertor->convertType(scalar.getType()));
    // result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::DivScalarOp>(op, resultType, input, scalar);


    return mlir::success();

}

//TODO: Replace the current decomposition with constOp and viewOp 
template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenOnesOp>::matchAndRewrite(
    mlir::torch::Torch::AtenOnesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value dtype = op.getOperand(1);
    Value layout = op.getOperand(2);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    input.setType(convertor->convertType(input.getType()));
    dtype.setType(convertor->convertType(dtype.getType()));
    layout.setType(convertor->convertType(layout.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::OnesOp>(op, resultType, input, dtype, layout);     
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenPowTensorScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenPowTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value exponent = op.getOperand(1);
    MLIRContext *context = getContext();
    
    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    input.setType(convertor->convertType(input.getType()));
    exponent.setType(convertor->convertType(exponent.getType()));
    // result.setType(resultType);
    
    rewriter.replaceOpWithNewOp<vllm_graph::PowOp>(op, resultType, input, exponent);


    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenAddTensorOp>::matchAndRewrite(
    mlir::torch::Torch::AtenAddTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Tensor1 = op.getOperand(0);
    Value Tensor2 = op.getOperand(1);
    Value Alpha = op.getOperand(2);

    //converting from torch.int or torch.float to simply int32, float32 
    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    Tensor1.setType(convertor->convertType(Tensor1.getType()));
    Tensor2.setType(convertor->convertType(Tensor2.getType()));
    Alpha.setType(convertor->convertType(Alpha.getType()));

    assert(Tensor1.getType() == Tensor2.getType() && "The dtypes of the tensors1 must match");

    rewriter.replaceOpWithNewOp<vllm_graph::AddOp>(op, resultType, Tensor1, Tensor2, Alpha);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenUnsqueezeOp>::matchAndRewrite(
    mlir::torch::Torch::AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value dim = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));

    dim.setType(convertor->convertType(dim.getType()));    

    rewriter.replaceOpWithNewOp<vllm_graph::UnsqueezeOp>(op, resultType, input, dim);

    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenSqueezeDimOp>::matchAndRewrite(
    mlir::torch::Torch::AtenSqueezeDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value dim = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());

    input.setType(convertor->convertType(input.getType()));

    dim.setType(convertor->convertType(dim.getType()));    
    // result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::SqueezeOp>(op, resultType, input, dim);

    return mlir::success();

}


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenAddScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenAddScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Operand1 = op.getOperand(0);
    Value Operand2 = op.getOperand(1);
    Value Alpha = op.getOperand(2);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());

    Operand1.setType(convertor->convertType(Operand1.getType()));
    Operand2.setType(convertor->convertType(Operand2.getType()));
    Alpha.setType(convertor->convertType(Alpha.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::AddOp>(op, resultType, Operand1, Operand2, Alpha);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenRsubScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenRsubScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Operand1 = op.getOperand(0);
    Value Operand2 = op.getOperand(1);
    Value Alpha = op.getOperand(2);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);

    Operand1.setType(convertor->convertType(Operand1.getType()));
    Operand2.setType(convertor->convertType(Operand2.getType()));
    Alpha.setType(convertor->convertType(Alpha.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::SubOp>(op, resultType, Operand2, Operand1, Alpha);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMulScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMulScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    
    Value Operand1 = op.getOperand(0);
    Value Operand2 = op.getOperand(1);

    Value result = op.getResult();
    const TypeConverter *convertor = getTypeConverter();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);

    if(!isa<mlir::torch::Torch::IntType>(Operand2.getType()) || isa<mlir::torch::Torch::FloatType>(Operand2.getType()))
        assert(false && "Scalar must be integer or float");

    Operand1.setType(convertor->convertType(Operand1.getType()));
    Operand2.setType(convertor->convertType(Operand2.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::MulOp>(op, resultType, Operand1, Operand2);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMulTensorOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMulTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    
    Value Tensor1 = op.getOperand(0);
    Value Tensor2 = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);

    Tensor1.setType(convertor->convertType(Tensor1.getType()));
    Tensor2.setType(convertor->convertType(Tensor2.getType()));

    assert(Tensor1.getType() == Tensor2.getType() && "The dtypes of the tensors1 must match");

    rewriter.replaceOpWithNewOp<vllm_graph::MulOp>(op, resultType, Tensor1, Tensor2);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenTransposeIntOp>::matchAndRewrite(
    mlir::torch::Torch::AtenTransposeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Input = op.getOperand(0);
    Value dim0 = op.getOperand(1);
    Value dim1 = op.getOperand(2);


    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());

    Input.setType(convertor->convertType(Input.getType()));
    dim0.setType(convertor->convertType(dim0.getType()));
    dim1.setType(convertor->convertType(dim1.getType()));

    // result.setType(resultType);
    rewriter.replaceOpWithNewOp<vllm_graph::TransposeOp>(op, resultType, Input, dim0, dim1);
    
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ValueTensorLiteralOp>::matchAndRewrite(
    mlir::torch::Torch::ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    Value Literalvalue = op.getResult();
    auto LiteralType = cast<mlir::torch::Torch::ValueTensorType>(Literalvalue.getType());
    assert(LiteralType && "Only Value tensor supported as of now");

    const TypeConverter *convertor = getTypeConverter();
    Type LiteralTensorType = convertor->convertType(Literalvalue.getType());

    // Literalvalue.setType(LiteralTensorType);
    DenseElementsAttr values = mlir::cast<DenseElementsAttr>(adaptor.getValue());

    if(values && values.getElementType().isF64()){
        SmallVector<float, 16> floatValues;
        bool losesInfo;
        for (auto val : values.getValues<APFloat>()) {
            APFloat floatVal = val; // Copy
            floatVal.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
            floatValues.push_back(floatVal.convertToFloat());
        }

        if(losesInfo){
            llvm::errs() << "Warning: Precision loss or rounding occurred during conversion.\n";
        }
        // Create the new f32 type
        ArrayRef<float> floatArray(floatValues.begin(), floatValues.end());
        ArrayRef<int64_t> shape(LiteralType.getOptionalSizes()->begin(), LiteralType.getOptionalSizes()->end());
        auto f32Type = RankedTensorType::get(shape, rewriter.getF32Type());
        values = DenseElementsAttr::get(f32Type, floatArray);

    }
    rewriter.replaceOpWithNewOp<vllm_graph::ValueTensorLiteralOp>(op, LiteralTensorType, values);
    return success();
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMatmulOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMatmulOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    Value input = op.getOperand(0);
    Value weight = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));
    weight.setType(convertor->convertType(weight.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::MatmulOp>(op, resultType, input, weight);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMmOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMmOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    Value input = op.getOperand(0);
    Value weight = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));
    weight.setType(convertor->convertType(weight.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::MatmulOp>(op, resultType, input, weight);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenBmmOp>::matchAndRewrite(
    mlir::torch::Torch::AtenBmmOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value weight = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));
    weight.setType(convertor->convertType(weight.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::BMMOp>(op, resultType, input, weight);
    return mlir::success();

    }

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenSoftmaxIntOp>::matchAndRewrite(
    mlir::torch::Torch::AtenSoftmaxIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    MLIRContext *context = op.getContext();

    Value input = op.getOperand(0);
    Value dim = op.getOperand(1);
    Value result = op.getResult();

    const TypeConverter *convertor = getTypeConverter();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));
    dim.setType(convertor->convertType(dim.getType()));

    rewriter.replaceOpWithNewOp<vllm_graph::SoftmaxOp>(op, resultType, input, dim);

    return success();
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::Aten_SoftmaxOp>::matchAndRewrite(
    mlir::torch::Torch::Aten_SoftmaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    

    MLIRContext *context = op.getContext();

    Value input = op.getOperand(0);
    Value dim = op.getOperand(1);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    // result.setType(resultType);
    input.setType(convertor->convertType(input.getType()));
    dim.setType(convertor->convertType(dim.getType()));
    
    rewriter.replaceOpWithNewOp<vllm_graph::SoftmaxOp>(op, resultType, input, dim);
    return success();
}


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenViewOp>::matchAndRewrite(
    mlir::torch::Torch::AtenViewOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

        Value input = op.getOperand(0);
        Value shape = op.getOperand(1);

        const TypeConverter *convertor = getTypeConverter();
        Value result = op.getResult();
        Type resultType = convertor->convertType(op.getResult().getType());

        input.setType(convertor->convertType(input.getType()));
        shape.setType(convertor->convertType(shape.getType()));

        rewriter.replaceOpWithNewOp<vllm_graph::ViewOp>(op, resultType, input, shape);
        
        return success();
        
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenScaledDotProductAttentionOp>::matchAndRewrite(
    mlir::torch::Torch::AtenScaledDotProductAttentionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

        Value query = op.getOperand(0);
        Value key = op.getOperand(1);
        Value value = op.getOperand(2);
        Value attn_mask = op.getOperand(3);
        Value dropout_p = op.getOperand(4);
        Value is_causal = op.getOperand(5);
        Value scale = op.getOperand(6);
        Value enable_gqa = op.getOperand(7);


        const TypeConverter *convertor = getTypeConverter();
        Value result = op.getResult();
        Type resultType = convertor->convertType(op.getResult().getType());

        query.setType(convertor->convertType(query.getType()));
        key.setType(convertor->convertType(key.getType()));
        value.setType(convertor->convertType(value.getType()));
        attn_mask.setType(convertor->convertType(attn_mask.getType()));
        dropout_p.setType(convertor->convertType(dropout_p.getType()));
        is_causal.setType(convertor->convertType(is_causal.getType()));
        scale.setType(convertor->convertType(scale.getType()));
        enable_gqa.setType(convertor->convertType(enable_gqa.getType()));

        rewriter.replaceOpWithNewOp<vllm_graph::ScaledDotProductAttentionOp>(op, resultType, 
                                                                            query,
                                                                            key,
                                                                            value,
                                                                            attn_mask,
                                                                            dropout_p,
                                                                            is_causal,
                                                                            scale,
                                                                            enable_gqa);
        
        return success();
        
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenWhereSelfOp>::matchAndRewrite(
    mlir::torch::Torch::AtenWhereSelfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter ) const {

        Value condition = op.getOperand(0);
        Value trueInput = op.getOperand(1);
        Value falseInput = op.getOperand(2);

        const TypeConverter *convertor = getTypeConverter();
        Value result = op.getResult();
        Type resultType = convertor->convertType(op.getResult().getType());
        // result.setType(resultType);
        condition.setType(convertor->convertType(condition.getType()));
        trueInput.setType(convertor->convertType(trueInput.getType()));
        falseInput.setType(convertor->convertType(falseInput.getType()));

        rewriter.replaceOpWithNewOp<vllm_graph::WhereOp>(op, resultType, condition, trueInput, falseInput);
        return success();
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::PrimListConstructOp>::matchAndRewrite(
    mlir::torch::Torch::PrimListConstructOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    OperandRange operandRange = op.getOperands();

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    for(auto Operand : operandRange){
        Operand.setType(convertor->convertType(Operand.getType()));
    }

    ValueRange newValueRange(operandRange);
    rewriter.replaceOpWithNewOp<vllm_graph::ListOp>(op, resultType, newValueRange);
    return success();

}


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenEmbeddingOp>::matchAndRewrite(
    mlir::torch::Torch::AtenEmbeddingOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value weight = op.getOperand(0);
    Value input = op.getOperand(1);
    MLIRContext *context = getContext();

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());

    weight.setType(convertor->convertType(weight.getType()));
    input.setType(convertor->convertType(input.getType()));
    // result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::EmbeddingOp>(op, resultType, input, weight);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenSliceTensorOp>::matchAndRewrite(
    mlir::torch::Torch::AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    MLIRContext *context = getContext();
    Value self = op.getOperand(0);
    Value dim = op.getOperand(1);
    Value start = op.getOperand(2);
    Value end = op.getOperand(3);
    Value step = op.getOperand(4);

    Location loc = op.getLoc();

    int64_t start_index = start.getDefiningOp<torch::Torch::ConstantIntOp>().getValue();
    int64_t end_index = end.getDefiningOp<torch::Torch::ConstantIntOp>().getValue();
    int64_t step_size = step.getDefiningOp<torch::Torch::ConstantIntOp>().getValue();

    std::vector<int32_t> range;
    for(int32_t i = start_index; i < end_index; i+=step_size)
        range.push_back(i);
    
    ArrayRef<int32_t> range_array(range.data(), range.size());
    int64_t x = static_cast<int64_t>(range.size());
    int64_t shape[] = {x};
    Type elemType = rewriter.getIntegerType(32);

    auto IndicesType = vllm_graph::ValueTensorType::get(context, ArrayRef<int64_t>(shape, 1), elemType);//RankedTensorType::get(ArrayRef<int64_t>(shape, 1), elemType);
    
    ShapedType shapetype = RankedTensorType::get(ArrayRef<int64_t>(shape, 1), elemType);
    auto denseAttr = DenseElementsAttr::get(shapetype, range_array);

    Value constRangeValOp = rewriter.create<vllm_graph::ValueTensorLiteralOp>(loc, IndicesType, denseAttr);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
    self.setType(convertor->convertType(self.getType()));
    dim.setType(convertor->convertType(dim.getType()));
    // result.setType(resultType);
    
    Value indexSelectResult = rewriter.create<vllm_graph::IndexSelectOp>(loc, resultType, self, dim, constRangeValOp);
    result.replaceAllUsesWith(indexSelectResult);
    rewriter.eraseOp(cast<Operation*>(op));

    return mlir::success();
    

}
template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenNativeLayerNormOp>::matchAndRewrite(
    mlir::torch::Torch::AtenNativeLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value inputArg = op.getOperand(0);
    Value normalisedShape = op.getOperand(1);
    Value weight = op.getOperand(2);
    Value bias = op.getOperand(3);
    Value epsilon = op.getOperand(4);

    Location loc = op.getLoc();

    
    Value OldResult = op.getResult(0);
    Value result = op.getResult(0);
    const TypeConverter *convertor = getTypeConverter();
    Type resultType = convertor->convertType(op.getResult(0).getType());

    inputArg.setType(convertor->convertType(inputArg.getType()));
    normalisedShape.setType(convertor->convertType(normalisedShape.getType()));
    weight.setType(convertor->convertType(weight.getType()));
    bias.setType(convertor->convertType(bias.getType()));
    epsilon.setType(convertor->convertType(epsilon.getType()));

    // result.setType(resultType);
    vllm_graph::LayerNormOp layerNormOp = rewriter.create<vllm_graph::LayerNormOp>(loc, resultType, inputArg, normalisedShape, weight, bias, epsilon);
    OldResult.replaceAllUsesWith(layerNormOp.getResult());
    rewriter.eraseOp(cast<Operation*>(op));
    return success();
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenBroadcastToOp>::matchAndRewrite(
    mlir::torch::Torch::AtenBroadcastToOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

        Value input = op.getOperand(0);
        Value shape = op.getOperand(1);

        Value result = op.getResult();

        const TypeConverter *convertor = getTypeConverter();
        Type resultType = convertor->convertType(op.getResult().getType());
        // result.setType(resultType);
        input.setType(convertor->convertType(input.getType()));
        shape.setType(convertor->convertType(shape.getType()));
       
        rewriter.replaceOpWithNewOp<vllm_graph::BroadCastOp>(op, resultType, input, shape);

        return success();

}

template<>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenToDtypeOp>::matchAndRewrite(
    mlir::torch::Torch::AtenToDtypeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter ) const {


        Value input = op.getOperand(0);
        Value dtype = op.getOperand(1);

        const TypeConverter *convertor = getTypeConverter();

        int dtype_val = dtype.getDefiningOp<torch::Torch::ConstantIntOp>().getValue();

        Value result = op.getResult();

        auto resultType = cast<mlir::torch::Torch::ValueTensorType>(result.getType());
        Type elemType = resultType.getOptionalDtype();

        input.setType(convertor->convertType(input.getType()));
        dtype.setType(convertor->convertType(dtype.getType()));
        

        //Erasing Op when float to float conversion 
        if(isa<mlir::torch::Torch::FloatType>(elemType) && dtype_val == 6){
            result.replaceAllUsesWith(input);
            rewriter.eraseOp(cast<Operation*>(op));
        } else {
            const TypeConverter *convertor = getTypeConverter();
            Type resultType = convertor->convertType(op.getResult().getType());
            rewriter.replaceOpWithNewOp<vllm_graph::DtypeCastOp>(op, resultType, input, dtype);
        }

        return success();

}


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenAddmmOp>::matchAndRewrite(
    mlir::torch::Torch::AtenAddmmOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

        
        Value bias = op.getOperand(0);
        Value input = op.getOperand(1);
        Value weight = op.getOperand(2);
        Value Alpha = op.getOperand(3);
        Value beta = op.getOperand(4);

        Value result = op.getResult();
        const TypeConverter *convertor = getTypeConverter();
        Type resultType = convertor->convertType(op.getResult().getType());
        // result.setType(resultType);

        bias.setType(convertor->convertType(bias.getType()));
        input.setType(convertor->convertType(input.getType()));
        weight.setType(convertor->convertType(weight.getType()));
        Alpha.setType(convertor->convertType(Alpha.getType()));
        beta.setType(convertor->convertType(beta.getType()));

        rewriter.replaceOpWithNewOp<vllm_graph::AddmmOp>(op, resultType, bias, input, weight, Alpha, beta);
        
        return success();
        

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenPermuteOp>::matchAndRewrite(
    mlir::torch::Torch::AtenPermuteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

        Value input = op.getOperand(0);
        Value shape = op.getOperand(1);

        Value result = op.getResult();
        const TypeConverter *convertor = getTypeConverter();
        Type resultType = convertor->convertType(op.getResult().getType());

        input.setType(convertor->convertType(input.getType()));
        shape.setType(convertor->convertType(shape.getType()));
        // result.setType(resultType);
        rewriter.replaceOpWithNewOp<vllm_graph::PermuteOp>(op, resultType, input, shape);

        return success();
        

}



template <>
LogicalResult EraseOp<mlir::torch::Torch::AtenDropoutOp>::matchAndRewrite(
    mlir::torch::Torch::AtenDropoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
        MLIRContext *context = getContext();
        Value operand = op.getOperand(0);

        const TypeConverter *convertor = getTypeConverter();
        operand.setType(convertor->convertType(operand.getType()));
        Value result = op.getResult();

        result.replaceAllUsesWith(operand);
        rewriter.eraseOp(cast<Operation*>(op));
        return success();
    }
} //namespace 



namespace {

class vLLMGraphConversion : public TypeConverter {
private:

public:
    vLLMGraphConversion(MLIRContext *context) {
        // Add conversions for primitive types
        
        addConversion([](Type type) -> std::optional<Type> {
            // Pass-through unchanged types
            return type;
        });

        // Convert f32 to f64
        addConversion([this, context](torch::Torch::ValueTensorType type) -> std::optional<Type> {
            return convertTorchvTypeTovLLMvType(type, context);      
        });

        addConversion([context](torch::Torch::FloatType type) -> std::optional<Type> {
            return Float32Type::get(context);
        });

        addConversion([context](torch::Torch::IntType type) -> std::optional<Type> {
            return IntegerType::get(context, 32);
        });

        addConversion([context](torch::Torch::BoolType type) -> std::optional<Type> {
            return IntegerType::get(context, 1);
        });

        addConversion([context](torch::Torch::ListType type) -> std::optional<Type> {
            auto containedType = convertvLLMContainedType(type, context);
            return vllm_graph::ListType::get(context, containedType);
        });

        addConversion([context](torch::Torch::NoneType type) -> std::optional<Type> {
            return vllm_graph::NoneType::get(context);
        });

        addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {

            return builder.create<vllm_graph::CastOp>(loc, type, inputs[0]).getResult();
        });

        addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {

            return builder.create<vllm_graph::CastOp>(loc, type, inputs[0]).getResult();
        });


    }

};


class ConvertTorchTovLLMGraph : public ConvertTorchTovLLMGraphBase<ConvertTorchTovLLMGraph> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<vllm_graph::vLLMGraphIRDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<vllm_graph::vLLMGraphIRDialect, arith::ArithDialect, func::FuncDialect>();

        vLLMGraphConversion typeConverter(context);

        target.addIllegalDialect<mlir::torch::Torch::TorchDialect>();

        RewritePatternSet patterns(context);
        //target.addLegalOp<mlir::torch::Torch::ConstantNoneOp>();

        target.addIllegalOp<mlir::torch::Torch::AtenReluOp>();                                               
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenReluOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::ConstantIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::ConstantNoneOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantNoneOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::ConstantFloatOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantFloatOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::ConstantBoolOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantBoolOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenAddTensorOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenAddTensorOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenRsubScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenRsubScalarOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenAddScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenAddScalarOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenScaledDotProductAttentionOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenScaledDotProductAttentionOp>>(typeConverter,
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::ConstantDeviceOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantDeviceOp>>(typeConverter,
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenOnesOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenOnesOp>>(typeConverter,
                                                         context);


        target.addIllegalOp<mlir::torch::Torch::AtenMulTensorOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenMulTensorOp>>(typeConverter,        
                                                         context);                                                    

        target.addIllegalOp<mlir::torch::Torch::AtenMulScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenMulScalarOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenUnsqueezeOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenUnsqueezeOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::AtenSqueezeDimOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenSqueezeDimOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenTransposeIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenTransposeIntOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::ValueTensorLiteralOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ValueTensorLiteralOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenWhereSelfOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenWhereSelfOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::AtenSliceTensorOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenSliceTensorOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenDivScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenDivScalarOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenMatmulOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenMatmulOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenMmOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenMmOp>>(typeConverter,        
                                                         context);
                
        target.addIllegalOp<mlir::torch::Torch::AtenBmmOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenBmmOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenSoftmaxIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenSoftmaxIntOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::Aten_SoftmaxOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::Aten_SoftmaxOp>>(typeConverter,        
                                                         context);


        target.addIllegalOp<mlir::torch::Torch::PrimListConstructOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::PrimListConstructOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenNativeLayerNormOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenNativeLayerNormOp>>(typeConverter,        
                                                         context);
            
        target.addIllegalOp<mlir::torch::Torch::AtenToDtypeOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenToDtypeOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenTanhOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenTanhOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenPowTensorScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenPowTensorScalarOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::AtenEmbeddingOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenEmbeddingOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenBroadcastToOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenBroadcastToOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenViewOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenViewOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenAddmmOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenAddmmOp>>(typeConverter, 
                                                         context);
                                                                
        target.addIllegalOp<mlir::torch::Torch::AtenPermuteOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenPermuteOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::AtenDropoutOp>();
        patterns.add<EraseOp<mlir::torch::Torch::AtenDropoutOp>>(typeConverter,        
                                                         context);
        

        if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
            return signalPassFailure();
    }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::vllm_graph::createTorchTovLLMGraph()
{
     return std::make_unique<ConvertTorchTovLLMGraph>();
}
