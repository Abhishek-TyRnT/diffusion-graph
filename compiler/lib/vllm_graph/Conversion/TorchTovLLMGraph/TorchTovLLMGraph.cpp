
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "../PassDetail.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
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
#include <iostream>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::vllm_graph;
using namespace mlir::torch::Torch;


Type convertTorchvTypeTovLLMvType(Type type, MLIRContext *context){

    auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    if(torchvTensor){
        Type opType;
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        torchvTensor.getOptionalDtype(), 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(vLLMvTensor);
        return opType;
    }

    else
        return type;
}


RankedTensorType convertTorchvTypeToTensorType(Type type){
    
    auto TorchTensor = cast<mlir::torch::Torch::ValueTensorType>(type);
    //assert(TorchTensor && "Only Value tensor supported as of now");
    Type elemType = TorchTensor.getOptionalDtype();
    SmallVector<int64_t> shape = cast<SmallVector<int64_t>>(TorchTensor.getOptionalSizes());
    
    RankedTensorType tensor = RankedTensorType::get(shape, elemType);
    return tensor;
}

Type convertvLLMContainedType(Type type, 
                        ConversionPatternRewriter &rewriter, 
                        MLIRContext *context){
    auto TorchList = cast<torch::Torch::ListType>(type);

    Type containedResultType;
    if(isa<torch::Torch::IntType>(TorchList.getContainedType()))
        containedResultType = rewriter.getIntegerType(32);
    else if(isa<torch::Torch::FloatType>(TorchList.getContainedType()))
        containedResultType = rewriter.getF32Type();
    else if(isa<torch::Torch::BoolType>(TorchList.getContainedType()))
        containedResultType = rewriter.getI1Type();
    else if(isa<torch::Torch::ValueTensorType>(TorchList.getContainedType())){
        containedResultType = convertTorchvTypeTovLLMvType(TorchList.getContainedType(), context);
    }
    else
        assert(false && "Type for the list not added");

    return containedResultType;
}

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

    Type opType = convertTorchvTypeTovLLMvType(selfTy, context);

    self.setType(opType);

    rewriter.replaceOpWithNewOp<vllm_graph::ReluOp>(op, opType, self);
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

    Type opType = convertTorchvTypeTovLLMvType(selfTy, context);

    self.setType(opType);

    rewriter.replaceOpWithNewOp<vllm_graph::TanhOp>(op, opType, self);
    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    int64_t constant = op.getValue();
    Type intType = rewriter.getIntegerType(32);
    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, constant, intType);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantBoolOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    bool constant = op.getValue();
    Type boolType = rewriter.getI1Type();
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
    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(op, constant, floatType);
    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenDivScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenDivScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value scalar = op.getOperand(1);
    Value result = op.getResult();
    MLIRContext *context = getContext();

    auto torchResultType = mlir::cast<torch::Torch::ValueTensorType>(op.getType());
    Type resultType = vllm_graph::ValueTensorType::get(context, torchResultType.getSizes(), torchResultType.getDtype());

    Type inputType = convertTorchvTypeTovLLMvType(input.getType(), context);
    input.setType(inputType);
    scalar.setType(rewriter.getF32Type());
    rewriter.replaceOpWithNewOp<vllm_graph::DivScalarOp>(op, resultType, input, scalar);


    return mlir::success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenPowTensorScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenPowTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value exponent = op.getOperand(1);
    Value result = op.getResult();
    MLIRContext *context = getContext();
    
    if(isa<mlir::torch::Torch::IntType>(exponent.getType()))
        exponent.setType(rewriter.getIntegerType(32));
    
    else if(isa<mlir::torch::Torch::FloatType>(exponent.getType()))
        exponent.setType(rewriter.getF32Type());
    else
        assert(false && "exponent must be integer or float");

    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);
    Type inputType = convertTorchvTypeTovLLMvType(input.getType(), context);
    input.setType(inputType);
    
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
    if(isa<mlir::torch::Torch::IntType>(Alpha.getType()))
        Alpha.setType(rewriter.getIntegerType(32));
    
    else if(isa<mlir::torch::Torch::FloatType>(Alpha.getType()))
        Alpha.setType(rewriter.getF32Type());
    else
        assert(false && "Alpha must be integer or float");

    Value result = op.getResult();

    MLIRContext *context = op.getContext();
    assert(Tensor1.getType() == Tensor2.getType() && "The dtypes of the tensors1 must match");

    Type Input1Type = convertTorchvTypeTovLLMvType(Tensor1.getType(), context);
    Type Input2Type = convertTorchvTypeTovLLMvType(Tensor2.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    // A special case where the input os coming from a literal tensor, meaning 
    // it is stored as constant, In vLLM Dialect the constants are stored as builtin MLIR types
    // Hence if it a constant then the input is RankedTensorType rather than vllm.vtensor. 
    if (auto *defOp = Tensor2.getDefiningOp()){
        if(defOp->getName() == mlir::OperationName("torch.vtensor.literal", op.getContext())){            
            mlir::torch::Torch::ValueTensorType valueTensor = cast<mlir::torch::Torch::ValueTensorType>(Input2Type);
            SmallVector<int64_t> sizes = cast<SmallVector<int64_t>>(valueTensor.getOptionalSizes());
            Type dtype = valueTensor.getOptionalDtype();
            RankedTensorType RankedInput = RankedTensorType::get(sizes, dtype);
            Input2Type = cast<Type>(RankedInput);
        }
    }
    Tensor1.setType(Input1Type);
    Tensor2.setType(Input2Type);

    rewriter.replaceOpWithNewOp<vllm_graph::AddOp>(op, resultType, Tensor1, Tensor2, Alpha);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenAddScalarOp>::matchAndRewrite(
    mlir::torch::Torch::AtenAddScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Operand1 = op.getOperand(0);
    Value Operand2 = op.getOperand(1);
    Value Alpha = op.getOperand(2);

    //converting from torch.int or torch.float to simply int32, float32 
    if(isa<mlir::torch::Torch::IntType>(Alpha.getType()))
        Alpha.setType(rewriter.getIntegerType(32));
    
    else if(isa<mlir::torch::Torch::FloatType>(Alpha.getType()))
        Alpha.setType(rewriter.getF32Type());
    else
        assert(false && "Alpha must be integer or float");

    Value result = op.getResult();

    MLIRContext *context = op.getContext();

    Type Input1Type = convertTorchvTypeTovLLMvType(Operand1.getType(), context);
    //Type Input2Type = convertTorchvTypeTovLLMvType(Operand2.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    if(isa<mlir::torch::Torch::IntType>(Operand2.getType()))
        Operand2.setType(rewriter.getIntegerType(32));
    
    else if(isa<mlir::torch::Torch::FloatType>(Operand2.getType()))
        Operand2.setType(rewriter.getF32Type());
    else
        assert(false && "Scalar must be integer or float");

    Operand1.setType(Input1Type);

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

    MLIRContext *context = op.getContext();

    Type Input1Type = convertTorchvTypeTovLLMvType(Operand1.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    if(isa<mlir::torch::Torch::IntType>(Operand2.getType()))
        Operand2.setType(rewriter.getIntegerType(32));
    
    else if(isa<mlir::torch::Torch::FloatType>(Operand2.getType()))
        Operand2.setType(rewriter.getF32Type());
    else
        assert(false && "Scalar must be integer or float");

    Operand1.setType(Input1Type);

    rewriter.replaceOpWithNewOp<vllm_graph::MulOp>(op, resultType, Operand1, Operand2);
    return mlir::success();
    
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMulTensorOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMulTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value Tensor1 = op.getOperand(0);
    Value Tensor2 = op.getOperand(1);

    Value result = op.getResult();

    MLIRContext *context = op.getContext();
    assert(Tensor1.getType() == Tensor2.getType() && "The dtypes of the tensors1 must match");

    Type Input1Type = convertTorchvTypeTovLLMvType(Tensor1.getType(), context);
    Type Input2Type = convertTorchvTypeTovLLMvType(Tensor2.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    // A special case where the input os coming from a literal tensor, meaning 
    // it is stored as constant, In vLLM Dialect the constants are stored as builtin MLIR types
    // Hence if it a constant then the input is RankedTensorType rather than vllm.vtensor. 
    if (auto *defOp = Tensor2.getDefiningOp()){
        if(defOp->getName() == mlir::OperationName("torch.vtensor.literal", op.getContext())){            
            mlir::torch::Torch::ValueTensorType valueTensor = cast<mlir::torch::Torch::ValueTensorType>(Input2Type);
            SmallVector<int64_t> sizes = cast<SmallVector<int64_t>>(valueTensor.getOptionalSizes());
            Type dtype = valueTensor.getOptionalDtype();
            RankedTensorType RankedInput = RankedTensorType::get(sizes, dtype);
            Input2Type = cast<Type>(RankedInput);
        }
    }
    Tensor1.setType(Input1Type);
    Tensor2.setType(Input2Type);

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

    //converting from torch.int to simply int32 
    if(isa<mlir::torch::Torch::IntType>(dim0.getType()))
        dim0.setType(rewriter.getIntegerType(32));
    else
        assert(false && "dim0 must be integer");
    
    if(isa<mlir::torch::Torch::IntType>(dim1.getType()))
        dim1.setType(rewriter.getIntegerType(32));
    else
        assert(false && "dim1 must be integer");    
    
    Value result = op.getResult();

    MLIRContext *context = op.getContext();

    Type InputType = convertTorchvTypeTovLLMvType(Input.getType(), context);

    // A special case where the input os coming from a literal tensor, meaning 
    // it is stored as constant, In vLLM Dialect the constants are stored as builtin MLIR types
    // Hence if it a constant then the input is RankedTensorType rather than vllm.vtensor. 
    if (auto *defOp = Input.getDefiningOp()){
        if(defOp->getName() == mlir::OperationName("torch.vtensor.literal", op.getContext())){            
            mlir::torch::Torch::ValueTensorType valueTensor = cast<mlir::torch::Torch::ValueTensorType>(InputType);
            SmallVector<int64_t> sizes = cast<SmallVector<int64_t>>(valueTensor.getOptionalSizes());
            Type dtype = valueTensor.getOptionalDtype();
            RankedTensorType RankedInput = RankedTensorType::get(sizes, dtype);
            InputType = cast<Type>(RankedInput);
        }
    }
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    Input.setType(InputType);

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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, LiteralTensorType, op.getValue());
    return success();
}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenMatmulOp>::matchAndRewrite(
    mlir::torch::Torch::AtenMatmulOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value weight = op.getOperand(1);

    Value result = op.getResult();
    MLIRContext *context = op.getContext();
    //TODO : Add dimension check

    Type inputType = convertTorchvTypeTovLLMvType(input.getType(), context);
    Type weightType = convertTorchvTypeTovLLMvType(weight.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);
    input.setType(inputType);
    weight.setType(weightType);

    rewriter.replaceOpWithNewOp<vllm_graph::MatmulOp>(op, resultType, input, weight);
    return mlir::success();

    }

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenBmmOp>::matchAndRewrite(
    mlir::torch::Torch::AtenBmmOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value input = op.getOperand(0);
    Value weight = op.getOperand(1);

    Value result = op.getResult();
    MLIRContext *context = op.getContext();
    //TODO : Add dimension check

    Type inputType = convertTorchvTypeTovLLMvType(input.getType(), context);
    Type weightType = convertTorchvTypeTovLLMvType(weight.getType(), context);
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);
    input.setType(inputType);
    weight.setType(weightType);

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
    Type inputType = convertTorchvTypeTovLLMvType(input.getType(), context);

    Value result = op.getResult();
    Type resultType = convertTorchvTypeTovLLMvType(result.getType(), context);

    input.setType(inputType);

    Value dim = op.getOperand(1);
    dim.setType(rewriter.getIntegerType(32));
    rewriter.replaceOpWithNewOp<vllm_graph::SoftmaxOp>(op, resultType, input, dim);

    return success();
}


template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::PrimListConstructOp>::matchAndRewrite(
    mlir::torch::Torch::PrimListConstructOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    OperandRange operandRange = op.getOperands();
    MLIRContext *context = getContext();
    for(Value operand : operandRange){
        Type operandType = operand.getType();
        Type updatedType;
        if(isa<torch::Torch::IntType>(operandType))
            updatedType = rewriter.getIntegerType(32);
        else if(isa<torch::Torch::FloatType>(operandType))
            updatedType = rewriter.getF32Type();
        else if(isa<torch::Torch::BoolType>(operandType))
            updatedType = rewriter.getI1Type();
        else if(isa<torch::Torch::ValueTensorType>(operandType)){
            updatedType = convertTorchvTypeTovLLMvType(operandType, context);
        }
        else
            assert(false && "Type for the list not added");

        operand.setType(updatedType);
    }

    auto containedResultType = convertvLLMContainedType(op.getResult().getType(), rewriter, context);
    auto resultType = vllm_graph::ListType::get(context, containedResultType);

    ValueRange newValueRange(operandRange);
    rewriter.replaceOpWithNewOp<vllm_graph::ListOp>(op, resultType, newValueRange);

    return success();

}

template <>
LogicalResult ConvertAtenOp<mlir::torch::Torch::AtenLayerNormOp>::matchAndRewrite(
    mlir::torch::Torch::AtenLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    Value inputArg = op.getOperand(0);
    Value normalisedShape = op.getOperand(1);
    Value weight = op.getOperand(2);
    Value bias = op.getOperand(3);
    Value epsilon = op.getOperand(4);

    MLIRContext *context = getContext();

    if(mlir::isa<torch::Torch::ConstantNoneOp>(*weight.getDefiningOp()) 
        || mlir::isa<torch::Torch::ConstantNoneOp>(*bias.getDefiningOp())
    )
        llvm::report_fatal_error("elementwise affine must be true\n");
    
    inputArg.setType(convertTorchvTypeTovLLMvType(inputArg.getType(), context));
    weight.setType(convertTorchvTypeToTensorType(weight.getType()));
    bias.setType(convertTorchvTypeToTensorType(bias.getType()));
    epsilon.setType(rewriter.getF32Type());
    auto containedType = convertvLLMContainedType(normalisedShape.getType(), rewriter, context);
    Type newNormalisedList = vllm_graph::ListType::get(context, containedType);
    normalisedShape.setType(newNormalisedList);
    
    Type resultType = convertTorchvTypeTovLLMvType(op.getResult().getType(), context);
    rewriter.replaceOpWithNewOp<vllm_graph::LayerNormOp>(op, resultType, inputArg, normalisedShape, weight, bias, epsilon);
    return success();
}

template <>
LogicalResult EraseOp<mlir::vllm_graph::CastOp>::matchAndRewrite(
    mlir::vllm_graph::CastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value operand = op.getOperand();
    Value result = op.getResult();
    mlir::torch::Torch::ValueTensorType operandType = mlir::cast<mlir::torch::Torch::ValueTensorType>(operand.getType());

    /*If the op is last op before return Op its operand type must be changed else might change 
    type of the func return */
    if(operandType){
        MLIRContext *context = op.getContext();
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        operandType.getOptionalSizes(), 
                        operandType.getOptionalDtype(), 
                        operandType.getOptionalSparsity());
        operand.setType(cast<Type>(vLLMvTensor));
    }

    result.replaceAllUsesWith(operand);
    rewriter.eraseOp(cast<Operation*>(op));
    return success();

    }
} //namespace 



namespace {
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

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

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

        target.addIllegalOp<mlir::torch::Torch::AtenTransposeIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenTransposeIntOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::ValueTensorLiteralOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ValueTensorLiteralOp>>(typeConverter,        
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
        
        target.addIllegalOp<mlir::vllm_graph::CastOp>();
        patterns.add<EraseOp<mlir::vllm_graph::CastOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::AtenSoftmaxIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenSoftmaxIntOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::PrimListConstructOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::PrimListConstructOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenLayerNormOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenLayerNormOp>>(typeConverter,        
                                                         context);
        target.addIllegalOp<mlir::torch::Torch::AtenTanhOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenTanhOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenPowTensorScalarOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenPowTensorScalarOp>>(typeConverter,        
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
