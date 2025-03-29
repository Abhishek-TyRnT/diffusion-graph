
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

using namespace mlir;
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
    result.setType(resultType);

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
    result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::TanhOp>(op, resultType, self);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
    int64_t constant = op.getValue();
    Type intType = rewriter.getIntegerType(32);
    Value result = op.getResult();
    result.setType(intType);
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
    result.setType(boolType);

    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, boolType, rewriter.getBoolAttr(constant));
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
    result.setType(floatType);

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
    result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::DivScalarOp>(op, resultType, input, scalar);


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
    result.setType(resultType);
    
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
    result.setType(resultType);


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
    result.setType(resultType);

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
    result.setType(resultType);

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
    result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::AddOp>(op, resultType, Operand1, Operand2, Alpha);
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
    result.setType(resultType);

    if(!isa<mlir::torch::Torch::IntType>(Operand2.getType()) || isa<mlir::torch::Torch::FloatType>(Operand2.getType()))
        assert(false && "Scalar must be integer or float");


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
    result.setType(resultType);

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
    result.setType(resultType);
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
    Type elemType = LiteralType.getOptionalDtype();
    SmallVector<int64_t> shape = cast<SmallVector<int64_t>>(LiteralType.getOptionalSizes());
    
    RankedTensorType LiteralTensorType = RankedTensorType::get(shape, elemType);
    Literalvalue.setType(LiteralTensorType);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, LiteralTensorType, adaptor.getValue());
    
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
    result.setType(resultType);

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
    result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::BMMOp>(op, resultType, input, weight);
    return mlir::success();

    }

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenSoftmaxIntOp>::matchAndRewrite(
    mlir::torch::Torch::AtenSoftmaxIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    MLIRContext *context = op.getContext();


    //Erasing dtype operand
    Value dtype = op.getOperand(2);
    auto dtypeOp = dtype.getDefiningOp();
    rewriter.eraseOp(cast<Operation*>(dtypeOp));


    Value input = op.getOperand(0);
    Value dim = op.getOperand(1);
    Value result = op.getResult();

    const TypeConverter *convertor = getTypeConverter();
    Type resultType = convertor->convertType(op.getResult().getType());
    result.setType(resultType);

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
    result.setType(resultType);

    
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
        result.setType(resultType);
        rewriter.replaceOpWithNewOp<vllm_graph::ViewOp>(op, resultType, input, shape);
        
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
    result.setType(resultType);

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
    result.setType(resultType);

    rewriter.replaceOpWithNewOp<vllm_graph::EmbeddingOp>(op, resultType, input, weight);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenSliceTensorOp>::matchAndRewrite(
    mlir::torch::Torch::AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

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

    RankedTensorType IndicesType = RankedTensorType::get(ArrayRef<int64_t>(shape, 1), elemType);
    
    auto denseAttr = DenseElementsAttr::get(IndicesType, range_array);
    Value constRangeValOp = rewriter.create<arith::ConstantOp>(loc, IndicesType, denseAttr);

    const TypeConverter *convertor = getTypeConverter();
    Value result = op.getResult();
    Type resultType = convertor->convertType(op.getResult().getType());
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
    result.setType(resultType);
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
        result.setType(resultType);
        rewriter.replaceOpWithNewOp<vllm_graph::BroadCastOp>(op, resultType, input, shape);

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
        result.setType(resultType);

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
        result.setType(resultType);
        rewriter.replaceOpWithNewOp<vllm_graph::PermuteOp>(op, resultType, input, shape);

        return success();
        

}


template <>
LogicalResult EraseOp<mlir::vllm_graph::CastOp>::matchAndRewrite(
    mlir::vllm_graph::CastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    
    Value operand = op.getOperand();
    Value result = op.getResult();
    auto operandType = mlir::cast<mlir::torch::Torch::ValueTensorType>(operand.getType());

    result.replaceAllUsesWith(operand);
    rewriter.eraseOp(cast<Operation*>(op));
    
    return success();

    }


template <>
LogicalResult EraseOp<mlir::torch::Torch::AtenDropoutOp>::matchAndRewrite(
    mlir::torch::Torch::AtenDropoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    
        MLIRContext *context = getContext();
        Value operand = op.getOperand(0);
        
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

        addConversion([context](torch::Torch::ListType type) -> std::optional<Type> {
            auto containedType = convertvLLMContainedType(type, context);
            return vllm_graph::ListType::get(context, containedType);
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
        target.addLegalOp<mlir::torch::Torch::ConstantNoneOp>();

        target.addIllegalOp<mlir::torch::Torch::AtenReluOp>();                                               
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenReluOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::ConstantIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>>(typeConverter,        
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

        target.addIllegalOp<mlir::torch::Torch::AtenAddScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenAddScalarOp>>(typeConverter,        
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

        target.addIllegalOp<mlir::torch::Torch::AtenSliceTensorOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenSliceTensorOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenDivScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenDivScalarOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenMatmulOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenMatmulOp>>(typeConverter,        
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

        //Erased operations
        target.addIllegalOp<mlir::vllm_graph::CastOp>();
        patterns.add<EraseOp<mlir::vllm_graph::CastOp>>(typeConverter,        
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
