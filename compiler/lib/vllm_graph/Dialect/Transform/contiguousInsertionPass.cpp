#include "llvm/Support/ErrorHandling.h"
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

using namespace mlir;
using namespace llvm;
using namespace mlir::diffusion_graph;
namespace {

template<typename FragmenterOp>
struct FragmentationPattern : public OpRewritePattern<FragmenterOp> {
    using OpRewritePattern<FragmenterOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(FragmenterOp op, PatternRewriter &rewriter) const override;
};


template<>
LogicalResult FragmentationPattern<diffusion_graph::LayerNormOp>::matchAndRewrite(diffusion_graph::LayerNormOp op, PatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    Value result = op.getResult();
    Value input = op.getOperand(0);
    
    auto viewOp = input.getDefiningOp<diffusion_graph::ViewOp>();
    if(!viewOp) return failure();
    
    Value viewResult = viewOp.getResult();
    auto contiguousOp = rewriter.create<diffusion_graph::ContiguousOp>(loc, viewResult.getType(), viewResult);
    op->setOperand(0, contiguousOp.getResult());
    return success();
}

template<>
LogicalResult FragmentationPattern<diffusion_graph::Conv2DOp>::matchAndRewrite(diffusion_graph::Conv2DOp op, PatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    Value result = op.getResult();
    Value input = op.getOperand(0);
    
    auto permuteOp = input.getDefiningOp<diffusion_graph::PermuteOp>();
    if(!permuteOp) return failure();
    
    Value permuteResult = permuteOp.getResult();
    auto contiguousOp = rewriter.create<diffusion_graph::ContiguousOp>(loc, permuteResult.getType(), permuteResult);
    op->setOperand(0, contiguousOp.getResult());
    return success();
}

template<>
LogicalResult FragmentationPattern<diffusion_graph::AddmmOp>::matchAndRewrite(diffusion_graph::AddmmOp op, PatternRewriter &rewriter) const {

    Location loc = op.getLoc();
    Value result = op.getResult();
    Value input = op.getOperand(1);
    // llvm::outs() << input << "\n";
    auto viewOp = input.getDefiningOp<diffusion_graph::ViewOp>();
    if(!viewOp) return failure();
    
    Value viewResult = viewOp.getResult();
    auto contiguousOp = rewriter.create<diffusion_graph::ContiguousOp>(loc, viewResult.getType(), viewResult);
    op->setOperand(1, contiguousOp.getResult());
    return success();
}

} //namespace

namespace {

struct ContiguousInsertionPass : 
    public mlir::diffusion_graph::ContiguousInsertionPassBase<ContiguousInsertionPass> {

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<diffusion_graph::DiffusionGraphIRDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
    }
    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();

        RewritePatternSet patterns(&getContext());

        // patterns.add<FragmentationPattern<diffusion_graph::LayerNormOp>>(&getContext());
        patterns.add<FragmentationPattern<diffusion_graph::Conv2DOp>>(&getContext());
        // patterns.add<FragmentationPattern<diffusion_graph::AddmmOp>>(&getContext());

        GreedyRewriteConfig config;
        config.maxIterations = 2;

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                            config))) {
            return signalPassFailure();
        }
    }
};

}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::diffusion_graph::createContiguousInsertionPass(){
    return std::make_unique<ContiguousInsertionPass>();
}

