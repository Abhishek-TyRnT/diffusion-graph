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
#include "vllm_graph/Dialect/IR/DiffusionGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "vllm_graph/Dialect/IR/DiffusionGraphTypes.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "PassDetail.hpp"

using namespace mlir;
using namespace llvm;
using namespace mlir::diffusion_graph;


namespace {

template<typename StaticOp>
struct StaticOpMaterializationPattern : public OpRewritePattern<StaticOp> {
    using OpRewritePattern<StaticOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(StaticOp op, PatternRewriter &rewriter) const override;
};



template<>
LogicalResult StaticOpMaterializationPattern<diffusion_graph::ArangeOp>::matchAndRewrite(diffusion_graph::ArangeOp op, PatternRewriter &rewriter) const {

    Value start = op.getOperands()[0];
    Value end = op.getOperands()[1];
    Value step = op.getOperands()[2];
    Value dtype = op.getOperands()[3];
    MLIRContext *context = op.getContext();

    if(start.getDefiningOp<arith::ConstantIntOp>() && 
       end.getDefiningOp<arith::ConstantIntOp>() && 
       step.getDefiningOp<arith::ConstantIntOp>() 
    ){
    
        int64_t start_index = start.getDefiningOp<arith::ConstantIntOp>().value();
        int64_t end_index = end.getDefiningOp<arith::ConstantIntOp>().value();
        int64_t step_size = step.getDefiningOp<arith::ConstantIntOp>().value();
        
        if(!dtype.getDefiningOp<arith::ConstantIntOp>())
            return rewriter.notifyMatchFailure(op, "dtype isn't mentioned\n");

    
        int64_t dtype_val = dtype.getDefiningOp<arith::ConstantIntOp>().value();

        if(dtype_val == 6){
        
            float start_val = static_cast<float>(start_index);
            float end_val = static_cast<float>(end_index);
            float step_val = static_cast<float>(step_size);
            std::vector<float> rangeVal;
            for(float i = start_val; i - end_val < 0.0f; i+=step_val){
                rangeVal.push_back(i);
            }

            ArrayRef<float> range_array(rangeVal.data(), rangeVal.size());
            int64_t size_arr[] = {static_cast<int64_t>(rangeVal.size())};
            auto RangeType = diffusion_graph::ValueTensorType::get(context, ArrayRef<int64_t>(size_arr, 1), rewriter.getF32Type());
        
            ShapedType shapetype = RankedTensorType::get(ArrayRef<int64_t>(size_arr, 1), rewriter.getF32Type());
            auto denseAttr = DenseElementsAttr::get(shapetype, range_array);

            rewriter.replaceOpWithNewOp<diffusion_graph::ValueTensorLiteralOp>(op, RangeType, denseAttr);
            
        } else if(dtype_val == 4) {
        
            std::vector<int32_t> rangeVal;
            for(int32_t i = start_index; i < end_index; i+=step_size){
                rangeVal.push_back(i);
            }

            ArrayRef<int32_t> range_array(rangeVal.data(), rangeVal.size());
            int64_t size_arr[] = {static_cast<int64_t>(rangeVal.size())};
            auto RangeType = diffusion_graph::ValueTensorType::get(context, ArrayRef<int64_t>(size_arr, 1), rewriter.getIntegerType(32));
        
            ShapedType shapetype = RankedTensorType::get(ArrayRef<int64_t>(size_arr, 1),rewriter.getIntegerType(32));
            auto denseAttr = DenseElementsAttr::get(shapetype, range_array);

            rewriter.replaceOpWithNewOp<diffusion_graph::ValueTensorLiteralOp>(op, RangeType, denseAttr);
        } else {
        
            return rewriter.notifyMatchFailure(op, "dtypes other than float32 or int32, not supported yet\n");
        }
        
    } else {
    
        return rewriter.notifyMatchFailure(op, "start, end, step, dtype must be constant\n");
    }



    return success();
}
} //namespace 

namespace {
struct StaticOpMaterialization
    : public mlir::diffusion_graph::StaticOpMaterializationPassBase<StaticOpMaterialization> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<diffusion_graph::DiffusionGraphIRDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<diffusion_graph::DiffusionGraphIRDialect, arith::ArithDialect, func::FuncDialect>();

    RewritePatternSet patterns(context);

    patterns.add<StaticOpMaterializationPattern<diffusion_graph::ArangeOp>>(context);

    GreedyRewriteConfig config;
    config.maxIterations = 2;

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                        config))) {
        return signalPassFailure();
    }
    
  }
};

} //namespace 

std::unique_ptr<OperationPass<func::FuncOp>> mlir::diffusion_graph::createStaticOpMaterializationPass(){
    return std::make_unique<StaticOpMaterialization>();
}



