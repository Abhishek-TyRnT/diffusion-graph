
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
#include "llvm/ADT/TypeSwitch.h"
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

    Type opType;
    if(auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(selfTy)){
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        torchvTensor.getOptionalDtype(), 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(vLLMvTensor);
    }

    Value new_op = rewriter.replaceOpWithNewOp<vllm_graph::ReluOp>(op, getTypeConverter()->convertType(selfTy), self);
    llvm::outs() << new_op << "\n";
    return success();

    }
}

class ConvertFuncOp : public OpConversionPattern<mlir::func::FuncOp> {
public:
  using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::func::FuncOp::Adaptor;
  LogicalResult
  matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    mlir::FunctionType oldFuncType = op.getFunctionType();
    llvm::outs() << op << "\n";

    // Get current argument and result types
    llvm::ArrayRef<mlir::Type> oldArgTypes = oldFuncType.getInputs();
    llvm::ArrayRef<mlir::Type> oldResultTypes = oldFuncType.getResults();

    
    llvm::SmallVector<mlir::Type, 4> newArgTypes;
    MLIRContext *context = op.getContext();

    auto typeConverter_fn = [context](Type type) {
    Type opType;
    if(auto torchvTensor = cast<mlir::torch::Torch::ValueTensorType>(type)){
        vllm_graph::ValueTensorType vLLMvTensor;
        vLLMvTensor = vLLMvTensor.get(context, 
                        torchvTensor.getOptionalSizes(), 
                        torchvTensor.getOptionalDtype(), 
                        torchvTensor.getOptionalSparsity());
        opType = cast<Type>(vLLMvTensor);
        return opType;
        }
    else
        return opType;
    };
    for (mlir::Type argType : oldArgTypes) {
        if(auto newargType = typeConverter_fn(argType))
            newArgTypes.push_back(newargType);
        else
            newArgTypes.push_back(argType);
    }

    llvm::SmallVector<mlir::Type, 4> newResultTypes;
    for (mlir::Type argType : oldResultTypes) {
        if(auto newargType = typeConverter_fn(argType))
            newResultTypes.push_back(newargType);
        else
            newResultTypes.push_back(argType);  
    }


    // Create the new function type with the updated argument types
    mlir::FunctionType newFuncType = rewriter.getFunctionType(newArgTypes, newResultTypes);
    auto newFuncOp = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), newFuncType);

    if (!op.getBody().empty()) {
      rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());

      // Now we need to replace the old arguments with the new arguments in the body
      mlir::Block &entryBlock = newFuncOp.front();
      for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
        mlir::Value oldArg = entryBlock.getArgument(i);
        mlir::Type newArgType = newArgTypes[i];

        // Create a new argument with the new type
        mlir::Value newArg = entryBlock.addArgument(newArgType, newFuncOp.getLoc());

        // Replace old argument uses with the new one
        oldArg.replaceAllUsesWith(newArg);

        // Optionally: Erase the old argument
        entryBlock.eraseArgument(i);
      }
    }

    // Erase the old function
    rewriter.eraseOp(op);

    // Set the new type on the function
    // rewriter.updateRootInPlace(op, [&]() {
    //   op.setType(newFuncType);
    // });

    // // If necessary, adjust the function body for the new argument types
    // mlir::Block &entryBlock = funcOp.front();
    // for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
    //   mlir::Value oldArg = entryBlock.getArgument(i);
    //   mlir::Type newArgType = newArgTypes[i];

    //   // Create a new argument with the new type
    //   mlir::Value newArg = entryBlock.addArgument(newArgType, funcOp.getLoc());

    //   // Replace old argument uses with the new one
    //   oldArg.replaceAllUsesWith(newArg);

    //   // Optionally: Erase the old argument
    //   entryBlock.eraseArgument(i);
    // }

    return mlir::success();

    }
};

namespace {
class ConvertTorchTovLLMGraph : public ConvertTorchTovLLMGraphBase<ConvertTorchTovLLMGraph> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<vllm_graph::vLLMGraphIRDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();

    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<vllm_graph::vLLMGraphIRDialect, arith::ArithDialect, func::FuncDialect>();

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        target.addIllegalDialect<mlir::torch::Torch::TorchDialect>();
        //llvm::outs() << getOperation() << "\n";

        RewritePatternSet patterns(context);
        target.addIllegalOp<mlir::torch::Torch::AtenReluOp>();                                               
        patterns.add<ConvertAtenOp<mlir::torch::Torch::AtenReluOp>>(typeConverter,        
                                                         context);
        patterns.add<ConvertFuncOp>(typeConverter,context);
        
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
