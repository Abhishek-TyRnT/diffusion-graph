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
#include <iostream>
using namespace mlir;
using namespace mlir::vllm_graph;
using namespace std;

//TODO: push these functions in utils file
bool hasStaticShape(ArrayRef<int64_t> shape){
    for(int64_t i : shape)
        if(i == DYNAMIC_SIZE)
            return false;
    
    return true;
}


namespace{
template<typename RootOp>
struct RecomposeSimpleOps : public OpRewritePattern<RootOp> {
    // LogicalResult match(RootOp op) const;
    // void rewrite(RootOp op, PatternRewriter &rewriter) const;
    using OpRewritePattern<RootOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(RootOp op, PatternRewriter &rewriter) const override;
};

bool isPatternUpsampleNearestFunc(vllm_graph::BroadCastIndexOp op, SmallVector<Operation*> &patternOpsToRecompose, vector<int64_t> &resizeDims){

    vllm_graph::ListOp IndexList = op.getOperands()[1].getDefiningOp<vllm_graph::ListOp>();

    if(!IndexList)
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(IndexList));

    OperandRange IndexListRange = IndexList.getOperands();


    if(IndexListRange.size() != 4)
        return false;

    //Batch dimension Pattern

    Value BatchDim = IndexListRange[0];

    if(!BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp BatchDimUnSqueezeOp1 = BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp1));

    BatchDim = BatchDimUnSqueezeOp1.getOperand(0);
    
    if(!BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp BatchDimUnSqueezeOp2 = BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp2));

    BatchDim = BatchDimUnSqueezeOp2.getOperand(0);

    
    if(!BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp BatchDimUnSqueezeOp3 = BatchDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp3));

    BatchDim = BatchDimUnSqueezeOp3.getOperand(0);


    
    if(!BatchDim.getDefiningOp<vllm_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDim.getDefiningOp<vllm_graph::ArangeOp>()));

    // Channel dimension pattern
    Value ChannelDim = IndexListRange[1];

    
    if(!ChannelDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp ChannelDimUnSqueezeOp1 = ChannelDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDimUnSqueezeOp1));

    ChannelDim = ChannelDimUnSqueezeOp1.getOperand(0);


    if(!ChannelDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp ChannelDimUnSqueezeOp2 = ChannelDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDimUnSqueezeOp2));

    ChannelDim = ChannelDimUnSqueezeOp2.getOperand(0);


    if(!ChannelDim.getDefiningOp<vllm_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDim.getDefiningOp<vllm_graph::ArangeOp>()));

    //Height dimension pattern
    Value HeightDim = IndexListRange[3];

    vllm_graph::ValueTensorType HeightDimValueTensorType = cast<vllm_graph::ValueTensorType>(HeightDim.getType());
    ArrayRef<int64_t> HeightDimShape = HeightDimValueTensorType.getSizes();

    // Dims not present, Can't work on dynamic dims yet
    if(HeightDimShape.size() == 0 || HeightDimShape[0] == -1)
        return false;

    resizeDims.push_back(HeightDimShape[0]);
    
    if(!HeightDim.getDefiningOp<vllm_graph::DtypeCastOp>())
        return false;

    vllm_graph::DtypeCastOp HeightDimDtypeCastOp = HeightDim.getDefiningOp<vllm_graph::DtypeCastOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimDtypeCastOp));

    HeightDim = HeightDimDtypeCastOp.getOperand(0);


    if(!HeightDim.getDefiningOp<vllm_graph::MulOp>())
        return false;

    vllm_graph::MulOp HeightDimMulOp = HeightDim.getDefiningOp<vllm_graph::MulOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimMulOp));

    HeightDim = HeightDimMulOp.getOperand(0);


    if(!HeightDim.getDefiningOp<vllm_graph::AddOp>())
        return false;

    vllm_graph::AddOp HeightDimAddOp = HeightDim.getDefiningOp<vllm_graph::AddOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimAddOp));

    HeightDim = HeightDimAddOp.getOperand(0);


    if(!HeightDim.getDefiningOp<vllm_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDim.getDefiningOp<vllm_graph::ArangeOp>()));

    //Width dimension pattern
    Value WidthDim = IndexListRange[2];

    vllm_graph::ValueTensorType WidthDimValueTensorType = cast<vllm_graph::ValueTensorType>(WidthDim.getType());
    ArrayRef<int64_t> WidthDimShape = WidthDimValueTensorType.getSizes();

    // Dims not present, Can't work on dynamic dims yet
    if(WidthDimShape.size() == 0 || WidthDimShape[0] == -1)
        return false;

    resizeDims.push_back(WidthDimShape[0]);

    if(!WidthDim.getDefiningOp<vllm_graph::UnsqueezeOp>())
        return false;

    vllm_graph::UnsqueezeOp WidthDimUnSqueezeOp = WidthDim.getDefiningOp<vllm_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimUnSqueezeOp));

    WidthDim = WidthDimUnSqueezeOp.getOperand(0);

    if(!WidthDim.getDefiningOp<vllm_graph::DtypeCastOp>())
        return false;

    vllm_graph::DtypeCastOp WidthDimDtypeCastOp = WidthDim.getDefiningOp<vllm_graph::DtypeCastOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimDtypeCastOp));
    WidthDim = WidthDimDtypeCastOp.getOperand(0);


    if(!WidthDim.getDefiningOp<vllm_graph::MulOp>())
        return false;

    vllm_graph::MulOp WidthDimMulOp = WidthDim.getDefiningOp<vllm_graph::MulOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimMulOp));

    WidthDim = WidthDimMulOp.getOperand(0);


    if(!WidthDim.getDefiningOp<vllm_graph::AddOp>())
        return false;

    vllm_graph::AddOp WidthDimAddOp = WidthDim.getDefiningOp<vllm_graph::AddOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimAddOp));

    WidthDim = WidthDimAddOp.getOperand(0);


    if(!WidthDim.getDefiningOp<vllm_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDim.getDefiningOp<vllm_graph::ArangeOp>()));

    
    return true;
}

template<>
LogicalResult RecomposeSimpleOps<vllm_graph::MatmulOp>::matchAndRewrite(vllm_graph::MatmulOp op, PatternRewriter &rewriter) const{

    Value res = op.getResult();
    MLIRContext *context = op.getContext();
    if(!res.hasOneUse())
        return rewriter.notifyMatchFailure(op, "Resulting op has many users, can't be fused");

    for(Operation *user_op : res.getUsers()){    
        if(!mlir::isa<vllm_graph::AddOp>(*user_op))
            return rewriter.notifyMatchFailure(op, "successor op is not an addOp");


        Value bias = user_op->getOperand(1);
        Value beta = user_op->getOperand(2);
        Value addRes = user_op->getResult(0);
        Location loc = op.getLoc();
        UnknownLoc unknownLoc = UnknownLoc::get(context);
        Type intType = rewriter.getIntegerType(32);
        //Seting alpha as one
        Value alpha = rewriter.create<arith::ConstantIntOp>(loc, 1, intType);

        Value input = op.getOperand(0);
        Value weight = op.getOperand(1);

        vllm_graph::ValueTensorType inputType = mlir::cast<vllm_graph::ValueTensorType>(input.getType());

        Type resultType = addRes.getType();
        Value dim0;
        Value dim1;
        Value dim2;

        Value size0;
        Value size1;
        Value size2;
        Value size3;

        if(!hasStaticShape(inputType.getSizes())){
            
            dim0 = rewriter.create<arith::ConstantIntOp>(loc, 0, intType );
            dim1 = rewriter.create<arith::ConstantIntOp>(loc, 1, intType );
            dim2 = rewriter.create<arith::ConstantIntOp>(loc, 2, intType );

        }

        ArrayRef<int64_t> input_shape = inputType.getSizes();
        ArrayRef<int64_t> result_shape =  mlir::cast<vllm_graph::ValueTensorType>(resultType).getSizes();
        // Need to reshape incase the input has batch size.
        if(input_shape.size() == 3){
            // c Array is Added for casting purposes
            if(hasStaticShape(inputType.getSizes())){
                int64_t new_input_size_c[] = {-1, input_shape[2]};
                ArrayRef<int64_t> viewInput_size(new_input_size_c, 2);
                auto vewInputType = vllm_graph::TupleType::get(context, rewriter.getIntegerType(64));
                auto DenseInputType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
                auto denseAttr = DenseElementsAttr::get(DenseInputType, viewInput_size);
                auto viewInputTupleOp = rewriter.create<vllm_graph::ConstTupleOp>(unknownLoc, vewInputType, denseAttr);
                
                int64_t new_result_view_size_c[] = {input_shape[0] * input_shape[1], input_shape[2]};
                ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto viewResultType = vllm_graph::ValueTensorType::get(context, viewResultSize, inputType.getDtype());

                input = rewriter.create<vllm_graph::ViewOp>(loc, viewResultType, input, viewInputTupleOp);

                int64_t new_result_size_c[] = {result_shape[0]*result_shape[1], result_shape[2]};
                ArrayRef<int64_t> ResultSize(new_result_size_c, 2);
                resultType = vllm_graph::ValueTensorType::get(context, ResultSize, inputType.getDtype());
            } else {
                size0 = rewriter.create<vllm_graph::SizeOp>(loc, intType, input, dim0);
                size1 = rewriter.create<vllm_graph::SizeOp>(loc, intType, input, dim1);
                size2 = rewriter.create<vllm_graph::SizeOp>(loc, intType, input, dim2);
                Value MulOp = rewriter.create<vllm_graph::MulOp>(loc, intType, size0, size1);
                Value value_array[] = {MulOp, size2}; 
                ArrayRef<Value> Operand_array(value_array, 2);
                ValueRange OperandList(Operand_array);
                Type viewListResultType = vllm_graph::ListType::get(context, intType);
                auto viewInputListOp = rewriter.create<vllm_graph::ListOp>(loc, viewListResultType, OperandList);
                
                int64_t new_result_view_size_c[] = {DYNAMIC_SIZE, input_shape[2]};
                ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto viewResultType = vllm_graph::ValueTensorType::get(context, viewResultSize, inputType.getDtype());
                
                input = rewriter.create<vllm_graph::ViewOp>(loc, viewResultType, input, viewInputListOp);
                 
            }
        }
        
        vllm_graph::AddmmOp addmmOp = rewriter.create<vllm_graph::AddmmOp>(loc, resultType, bias, input, weight, alpha, beta);
        Value new_res = addmmOp.getResult();
        if(input_shape.size() == 3){
            if(hasStaticShape(inputType.getSizes())){
                ArrayRef<int64_t> viewInput_size = result_shape;
                RankedTensorType DenseInputType = RankedTensorType::get({3}, rewriter.getIntegerType(64));
                auto denseAttr = DenseElementsAttr::get(DenseInputType, viewInput_size);

                auto vewInputType = vllm_graph::TupleType::get(context, rewriter.getIntegerType(64));
                auto viewInputTupleOp = rewriter.create<vllm_graph::ConstTupleOp>(unknownLoc, vewInputType, denseAttr);

                auto viewResultType = vllm_graph::ValueTensorType::get(context, viewInput_size, inputType.getDtype());
                new_res = rewriter.create<vllm_graph::ViewOp>(loc, viewResultType, new_res, viewInputTupleOp);
            } else {
                size3 = rewriter.create<vllm_graph::SizeOp>(loc, intType, weight, dim1);
                Value value_array[] = {size0, size1, size3}; 
                ArrayRef<Value> Operand_array(value_array, 3);
                ValueRange OperandList(Operand_array);

                Type viewListResultType = vllm_graph::ListType::get(context, intType);
                auto viewInputListOp = rewriter.create<vllm_graph::ListOp>(loc, viewListResultType, OperandList);
                
                // int64_t new_result_view_size_c[] = {DYNAMIC_SIZE, input_shape[2]};
                // ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto vllm_resultType = cast<vllm_graph::ValueTensorType>(resultType);
                auto viewResultType = vllm_graph::ValueTensorType::get(context, vllm_resultType.getSizes(), vllm_resultType.getDtype());
                
                new_res = rewriter.create<vllm_graph::ViewOp>(loc, viewResultType, new_res, viewInputListOp);
            }
            
        }

        addRes.replaceAllUsesWith(new_res);

        rewriter.eraseOp(user_op);
        rewriter.eraseOp(op);
        
    }

    return success();
}

template<> 
LogicalResult RecomposeSimpleOps<vllm_graph::BroadCastIndexOp>::matchAndRewrite(vllm_graph::BroadCastIndexOp op, PatternRewriter &rewriter) const{
    

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    SmallVector<Operation*> patternOpsToRecompose;
    vector<int64_t> resizeDims;
    if(!isPatternUpsampleNearestFunc(op, patternOpsToRecompose, resizeDims))
        return failure();


    // Lower to arith.constant with StringAttr - output is generic string
    llvm::StringRef str("nearest");
    auto stringAttr = rewriter.getStringAttr(str);
    auto constantStringOp = rewriter.create<vllm_graph::ConstantStringOp>(loc, stringAttr);
    
    ArrayRef<int64_t> resizeDimsSize(resizeDims.data(), 2);
    auto resizeDimsType = vllm_graph::TupleType::get(context, rewriter.getIntegerType(64));
    auto DenseInputType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
    auto denseAttr = DenseElementsAttr::get(DenseInputType, resizeDimsSize);
    auto resizeDimsTupleOp = rewriter.create<vllm_graph::ConstTupleOp>(loc, resizeDimsType, denseAttr);

    Value Input = op.getOperand(0);
    Value result = op.getResult();
    Type resultType = result.getType();
    
    rewriter.replaceOpWithNewOp<vllm_graph::UpsampleOp>(op, resultType, Input, resizeDimsTupleOp, constantStringOp);

    for ( Operation* op : patternOpsToRecompose){
        if(!op->hasSuccessors())
            rewriter.eraseOp(op);
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
        patterns.add<RecomposeSimpleOps<vllm_graph::BroadCastIndexOp>>(context);

        GreedyRewriteConfig config;
        config.useTopDownTraversal = true;
        config.maxIterations = GreedyRewriteConfig::kNoLimit;

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                            config))) {
            return signalPassFailure();
        }
    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::vllm_graph::createRecomposeSimpleOpsToComplexOps(){
    return std::make_unique<RecomposeSimpleOpsToComplex>();
}


