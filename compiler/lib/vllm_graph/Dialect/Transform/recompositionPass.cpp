
#include "mlir/IR/PatternMatch.h"   // For OpRewritePattern and PatternRewriter
#include "mlir/IR/Builders.h"       // For pattern rewriter utility
#include "mlir/IR/MLIRContext.h"    // MLIRContext
#include "mlir/IR/Operation.h"      // Operation
#include "mlir/Transforms/DialectConversion.h" // For RewritePatternSet
#include "mlir/Support/LogicalResult.h" // LogicalResult, success(), failure()
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "PassDetail.hpp"

using namespace mlir;
using namespace mlir::vllm_graph;

namespace{
template<typename RootOp>
struct RecomposeSimpleOps : public OpRewritePattern<RootOp> {
    // LogicalResult match(RootOp op) const;
    // void rewrite(RootOp op, PatternRewriter &rewriter) const;
    using OpRewritePattern<RootOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(RootOp op, PatternRewriter &rewriter) const override;
};

template<>
LogicalResult RecomposeSimpleOps<vllm_graph::MatmulOp>::matchAndRewrite(vllm_graph::MatmulOp op, PatternRewriter &rewriter) const{

    Value res = op.getResult();
    if(!res.hasOneUse())
        return rewriter.notifyMatchFailure(op, "Resulting op has many users, can't be fused");

    for(Operation *user_op : res.getUsers()){    
        if(!mlir::isa<vllm_graph::AddOp>(*user_op))
            return rewriter.notifyMatchFailure(op, "successor op is not an addOp");

        //auto addOp = mlir::dyn_cast<vllm_graph::AddOp>(user_op);

        Value bias = user_op->getOperand(1);
        Value beta = user_op->getOperand(2);
        Location loc = op.getLoc();

        Type intType = rewriter.getIntegerType(32);
        //Seting alpha as one
        Value alpha = rewriter.create<arith::ConstantIntOp>(loc, 1, intType);

        Value input = op.getOperand(0);
        Value weight = op.getOperand(1);

        Type resultType = input.getType();
        vllm_graph::AddmmOp addmmOp = rewriter.create<vllm_graph::AddmmOp>(loc, resultType, bias, input, weight, alpha, beta);

        Value new_res = addmmOp.getResult();
        res.replaceAllUsesWith(new_res);

        rewriter.eraseOp(op);
        rewriter.eraseOp(user_op);
    }

    return success();
}
} //namespace


namespace{
class RecomposeSimpleOpsToComplex : public RecomposeSimpleOpsToComplexPassBase<RecomposeSimpleOpsToComplex> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<vllm_graph::vLLMGraphIRDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() override{
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<vllm_graph::vLLMGraphIRDialect, arith::ArithDialect, func::FuncDialect>();

        RewritePatternSet patterns(context);

        patterns.add<RecomposeSimpleOps<vllm_graph::MatmulOp>>(context);

        GreedyRewriteConfig config;
        config.useTopDownTraversal = true;
        config.maxIterations = GreedyRewriteConfig::kNoLimit;

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
            return signalPassFailure();
        }
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::vllm_graph::createRecomposeSimpleOpsToComplexOps(){
    return std::make_unique<RecomposeSimpleOpsToComplex>();
}


