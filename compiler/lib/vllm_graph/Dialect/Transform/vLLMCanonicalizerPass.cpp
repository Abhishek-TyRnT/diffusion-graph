#include "mlir/IR/PatternMatch.h"   // For OpRewritePattern and PatternRewriter
#include "mlir/IR/Builders.h"       // For pattern rewriter utility
#include "mlir/IR/MLIRContext.h"    // MLIRContext
#include "mlir/IR/Operation.h"      // Operation
#include "mlir/IR/Location.h"
#include "mlir/Transforms/DialectConversion.h" // For RewritePatternSet
#include "mlir/Support/LogicalResult.h" // LogicalResult, success(), failure()
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "PassDetail.hpp"
#include <iostream>

using namespace mlir;
using namespace mlir::vllm_graph;

namespace {
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
LogicalResult EraseOp<vllm_graph::CastOp>::matchAndRewrite(
    vllm_graph::CastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    Value operand = op.getOperand();
    Value result = op.getResult();
    const TypeConverter *convertor = getTypeConverter();
    
    result.replaceAllUsesWith(operand);
    rewriter.eraseOp(cast<Operation*>(op));

    return success();
}

template <>
LogicalResult EraseOp<vllm_graph::BroadCastOp>::matchAndRewrite(
    vllm_graph::BroadCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    if(op.use_empty()){
        rewriter.eraseOp(cast<Operation*>(op));
    } else 
        return rewriter.notifyMatchFailure(op, "can't erase, not a dead op");

    return success();
}

template <>
LogicalResult EraseOp<vllm_graph::SizeOp>::matchAndRewrite(
    vllm_graph::SizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    if(op.use_empty()){
        rewriter.eraseOp(cast<Operation*>(op));
    } else 
        return rewriter.notifyMatchFailure(op, "can't erase, not a dead op");

    return success();
}

template <>
LogicalResult EraseOp<vllm_graph::DtypeCastOp>::matchAndRewrite(
    vllm_graph::DtypeCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
    
    if(op.use_empty()){
        rewriter.eraseOp(cast<Operation*>(op));
    } else 
        return rewriter.notifyMatchFailure(op, "can't erase, not a dead op");

    return success();
}

} //namespace

class vLLMGraphConversion : public TypeConverter {
private:

public:
    vLLMGraphConversion(MLIRContext *context) {
        // Add conversions for primitive types
        
        addConversion([](Type type) -> std::optional<Type> {
            // Pass-through unchanged types
            return type;
        });

        addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {

            return builder.create<vllm_graph::CastOp>(loc, type, inputs[0]).getResult();
        });

        addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {

            return builder.create<vllm_graph::CastOp>(loc, type, inputs[0]).getResult();
        });


    }

};

namespace {
class vLLMCanonicalizerPass : public vLLMCanonicalizerPassBase<vLLMCanonicalizerPass> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<vllm_graph::vLLMGraphIRDialect>();
        registry.insert<arith::ArithDialect>();
    }
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        vLLMGraphConversion typeConverter(context);
        RewritePatternSet patterns(context);
        ConversionTarget target(*context);
        // target.addILegalDialect<vllm_graph::vLLMGraphIRDialect, arith::ArithDialect, func::FuncDialect>();


        patterns.add<EraseOp<vllm_graph::CastOp>>(typeConverter,        
                                                         context);

        patterns.add<EraseOp<vllm_graph::BroadCastOp>>(typeConverter,        
                                                         context);
        
        patterns.add<EraseOp<vllm_graph::DtypeCastOp>>(typeConverter,        
                                                         context);

        patterns.add<EraseOp<vllm_graph::SizeOp>>(typeConverter,        
                                                         context);


        if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
            return signalPassFailure();
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::vllm_graph::createvLLMCanonicalizerPass(){
    return std::make_unique<vLLMCanonicalizerPass>();
}

