
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
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"
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
        return torchvTensor;
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
LogicalResult ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>::matchAndRewrite(
    mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

    int64_t constant = op.getValue();
    Type intType = rewriter.getIntegerType(32);
    Value arithOp = rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, constant, intType);
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

    Tensor1.setType(Input1Type);
    Tensor2.setType(Input2Type);

    rewriter.replaceOpWithNewOp<vllm_graph::AddOp>(op, resultType, Tensor1, Tensor2, Alpha);
    return mlir::success();
    
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
}



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
        target.addIllegalOp<mlir::torch::Torch::AtenReluOp>();                                               
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenReluOp>>(typeConverter,        
                                                         context);
        
        target.addIllegalOp<mlir::torch::Torch::ConstantIntOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::ConstantIntOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::torch::Torch::AtenAddTensorOp>();
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenAddTensorOp>>(typeConverter,        
                                                         context);

        target.addIllegalOp<mlir::vllm_graph::CastOp>();
        patterns.add<EraseOp<mlir::vllm_graph::CastOp>>(typeConverter,        
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
