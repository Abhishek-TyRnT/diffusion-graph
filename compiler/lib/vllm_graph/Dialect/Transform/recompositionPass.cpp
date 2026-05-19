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
using namespace mlir::diffusion_graph;
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

bool isPatternUpsampleNearestFunc(diffusion_graph::BroadCastIndexOp op, SmallVector<Operation*> &patternOpsToRecompose, vector<int64_t> &resizeDims){

    diffusion_graph::ListOp IndexList = op.getOperands()[1].getDefiningOp<diffusion_graph::ListOp>();

    if(!IndexList)
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(IndexList));

    OperandRange IndexListRange = IndexList.getOperands();


    if(IndexListRange.size() != 4)
        return false;

    //Batch dimension Pattern

    Value BatchDim = IndexListRange[0];

    if(!BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp BatchDimUnSqueezeOp1 = BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp1));

    BatchDim = BatchDimUnSqueezeOp1.getOperand(0);
    
    if(!BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp BatchDimUnSqueezeOp2 = BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp2));

    BatchDim = BatchDimUnSqueezeOp2.getOperand(0);

    
    if(!BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp BatchDimUnSqueezeOp3 = BatchDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDimUnSqueezeOp3));

    BatchDim = BatchDimUnSqueezeOp3.getOperand(0);


    
    if(!BatchDim.getDefiningOp<diffusion_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(BatchDim.getDefiningOp<diffusion_graph::ArangeOp>()));

    // Channel dimension pattern
    Value ChannelDim = IndexListRange[1];

    
    if(!ChannelDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp ChannelDimUnSqueezeOp1 = ChannelDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDimUnSqueezeOp1));

    ChannelDim = ChannelDimUnSqueezeOp1.getOperand(0);


    if(!ChannelDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp ChannelDimUnSqueezeOp2 = ChannelDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDimUnSqueezeOp2));

    ChannelDim = ChannelDimUnSqueezeOp2.getOperand(0);


    if(!ChannelDim.getDefiningOp<diffusion_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(ChannelDim.getDefiningOp<diffusion_graph::ArangeOp>()));

    //Height dimension pattern
    Value HeightDim = IndexListRange[3];

    diffusion_graph::ValueTensorType HeightDimValueTensorType = cast<diffusion_graph::ValueTensorType>(HeightDim.getType());
    ArrayRef<int64_t> HeightDimShape = HeightDimValueTensorType.getSizes();

    // Dims not present, Can't work on dynamic dims yet
    if(HeightDimShape.size() == 0 || HeightDimShape[0] == -1)
        return false;

    resizeDims.push_back(HeightDimShape[0]);
    
    if(!HeightDim.getDefiningOp<diffusion_graph::DtypeCastOp>())
        return false;

    diffusion_graph::DtypeCastOp HeightDimDtypeCastOp = HeightDim.getDefiningOp<diffusion_graph::DtypeCastOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimDtypeCastOp));

    HeightDim = HeightDimDtypeCastOp.getOperand(0);


    if(!HeightDim.getDefiningOp<diffusion_graph::MulOp>())
        return false;

    diffusion_graph::MulOp HeightDimMulOp = HeightDim.getDefiningOp<diffusion_graph::MulOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimMulOp));

    HeightDim = HeightDimMulOp.getOperand(0);


    if(!HeightDim.getDefiningOp<diffusion_graph::AddOp>())
        return false;

    diffusion_graph::AddOp HeightDimAddOp = HeightDim.getDefiningOp<diffusion_graph::AddOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDimAddOp));

    HeightDim = HeightDimAddOp.getOperand(0);


    if(!HeightDim.getDefiningOp<diffusion_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(HeightDim.getDefiningOp<diffusion_graph::ArangeOp>()));

    //Width dimension pattern
    Value WidthDim = IndexListRange[2];

    diffusion_graph::ValueTensorType WidthDimValueTensorType = cast<diffusion_graph::ValueTensorType>(WidthDim.getType());
    ArrayRef<int64_t> WidthDimShape = WidthDimValueTensorType.getSizes();

    // Dims not present, Can't work on dynamic dims yet
    if(WidthDimShape.size() == 0 || WidthDimShape[0] == -1)
        return false;

    resizeDims.push_back(WidthDimShape[0]);

    if(!WidthDim.getDefiningOp<diffusion_graph::UnsqueezeOp>())
        return false;

    diffusion_graph::UnsqueezeOp WidthDimUnSqueezeOp = WidthDim.getDefiningOp<diffusion_graph::UnsqueezeOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimUnSqueezeOp));

    WidthDim = WidthDimUnSqueezeOp.getOperand(0);

    if(!WidthDim.getDefiningOp<diffusion_graph::DtypeCastOp>())
        return false;

    diffusion_graph::DtypeCastOp WidthDimDtypeCastOp = WidthDim.getDefiningOp<diffusion_graph::DtypeCastOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimDtypeCastOp));
    WidthDim = WidthDimDtypeCastOp.getOperand(0);


    if(!WidthDim.getDefiningOp<diffusion_graph::MulOp>())
        return false;

    diffusion_graph::MulOp WidthDimMulOp = WidthDim.getDefiningOp<diffusion_graph::MulOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimMulOp));

    WidthDim = WidthDimMulOp.getOperand(0);


    if(!WidthDim.getDefiningOp<diffusion_graph::AddOp>())
        return false;

    diffusion_graph::AddOp WidthDimAddOp = WidthDim.getDefiningOp<diffusion_graph::AddOp>();

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDimAddOp));

    WidthDim = WidthDimAddOp.getOperand(0);


    if(!WidthDim.getDefiningOp<diffusion_graph::ArangeOp>())
        return false;

    patternOpsToRecompose.push_back(cast<Operation*>(WidthDim.getDefiningOp<diffusion_graph::ArangeOp>()));

    
    return true;
}

bool isPatternSDPA(diffusion_graph::TransposeOp op, SmallVector<Operation*> &patternOpsToErase, SmallVector<Value> &inputValues){
    
    Value input = op.getOperand(0);


    auto transpose2op = input.getDefiningOp<diffusion_graph::TransposeOp>();
    if(!transpose2op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(transpose2op));

    input = transpose2op.getOperand(0);

    auto view1Op = input.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view1Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view1Op));

    input = view1Op.getOperand(0);

    auto bmm1Op = input.getDefiningOp<diffusion_graph::BMMOp>();
    if(!bmm1Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(bmm1Op));

    Value value = bmm1Op.getOperand(1);

    auto view2Op = value.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view2Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view2Op));

    value = view2Op.getOperand(0);

    input = bmm1Op.getOperand(0);

    auto view3Op = input.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view3Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view3Op));

    input = view3Op.getOperand(0);

    auto softmax_op = input.getDefiningOp<diffusion_graph::SoftmaxOp>();
    if(!softmax_op)
        return false;

    patternOpsToErase.push_back(cast<Operation*>(softmax_op));

    input = softmax_op.getOperand(0);

    auto view4Op = input.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view4Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view4Op));

    input = view4Op.getOperand(0);

    auto bmm2Op = input.getDefiningOp<diffusion_graph::BMMOp>();
    if(!bmm2Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(bmm2Op));

    Value key = bmm2Op.getOperand(1);
    Value query = bmm2Op.getOperand(0);

    auto view5Op = key.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view5Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view5Op));

    key = view5Op.getOperand(0);
    auto Mul1Op = key.getDefiningOp<diffusion_graph::MulOp>();
    if(!Mul1Op)
        return false;

    patternOpsToErase.push_back(cast<Operation*>(Mul1Op));
    
    key = Mul1Op.getOperand(0);

    auto transpose3Op = key.getDefiningOp<diffusion_graph::TransposeOp>();
    if(!transpose3Op)
        return false;

    patternOpsToErase.push_back(cast<Operation*>(transpose3Op));

    key = transpose3Op.getOperand(0);

    auto view6Op = query.getDefiningOp<diffusion_graph::ViewOp>();
    if(!view6Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(view6Op));

    query = view6Op.getOperand(0);

    auto Mul2Op = query.getDefiningOp<diffusion_graph::MulOp>();
    if(!Mul2Op)
        return false;
    
    patternOpsToErase.push_back(cast<Operation*>(Mul2Op));

    query = Mul2Op.getOperand(0);
    
    inputValues.push_back(query);
    inputValues.push_back(key);
    inputValues.push_back(value);

    return true;
}

template<>
LogicalResult RecomposeSimpleOps<diffusion_graph::MatmulOp>::matchAndRewrite(diffusion_graph::MatmulOp op, PatternRewriter &rewriter) const{

    Value res = op.getResult();
    MLIRContext *context = op.getContext();
    if(!res.hasOneUse())
        return rewriter.notifyMatchFailure(op, "Resulting op has many users, can't be fused");

    for(Operation *user_op : res.getUsers()){    
        if(!mlir::isa<diffusion_graph::AddOp>(*user_op))
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

        diffusion_graph::ValueTensorType inputType = mlir::cast<diffusion_graph::ValueTensorType>(input.getType());

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
        ArrayRef<int64_t> result_shape =  mlir::cast<diffusion_graph::ValueTensorType>(resultType).getSizes();
        // Need to reshape incase the input has batch size.
        if(input_shape.size() == 3){
            // c Array is Added for casting purposes
            if(hasStaticShape(inputType.getSizes())){
                int64_t new_input_size_c[] = {-1, input_shape[2]};
                ArrayRef<int64_t> viewInput_size(new_input_size_c, 2);
                auto vewInputType = diffusion_graph::TupleType::get(context, rewriter.getIntegerType(64));
                auto DenseInputType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
                auto denseAttr = DenseElementsAttr::get(DenseInputType, viewInput_size);
                auto viewInputTupleOp = rewriter.create<diffusion_graph::ConstTupleOp>(unknownLoc, vewInputType, denseAttr);
                
                int64_t new_result_view_size_c[] = {input_shape[0] * input_shape[1], input_shape[2]};
                ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto viewResultType = diffusion_graph::ValueTensorType::get(context, viewResultSize, inputType.getDtype());

                input = rewriter.create<diffusion_graph::ViewOp>(loc, viewResultType, input, viewInputTupleOp);

                int64_t new_result_size_c[] = {result_shape[0]*result_shape[1], result_shape[2]};
                ArrayRef<int64_t> ResultSize(new_result_size_c, 2);
                resultType = diffusion_graph::ValueTensorType::get(context, ResultSize, inputType.getDtype());
            } else {
                size0 = rewriter.create<diffusion_graph::SizeOp>(loc, intType, input, dim0);
                size1 = rewriter.create<diffusion_graph::SizeOp>(loc, intType, input, dim1);
                size2 = rewriter.create<diffusion_graph::SizeOp>(loc, intType, input, dim2);
                Value MulOp = rewriter.create<diffusion_graph::MulOp>(loc, intType, size0, size1);
                Value value_array[] = {MulOp, size2}; 
                ArrayRef<Value> Operand_array(value_array, 2);
                ValueRange OperandList(Operand_array);
                Type viewListResultType = diffusion_graph::ListType::get(context, intType);
                auto viewInputListOp = rewriter.create<diffusion_graph::ListOp>(loc, viewListResultType, OperandList);
                
                int64_t new_result_view_size_c[] = {DYNAMIC_SIZE, input_shape[2]};
                ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto viewResultType = diffusion_graph::ValueTensorType::get(context, viewResultSize, inputType.getDtype());
                
                input = rewriter.create<diffusion_graph::ViewOp>(loc, viewResultType, input, viewInputListOp);
                 
            }
        }
        
        diffusion_graph::AddmmOp addmmOp = rewriter.create<diffusion_graph::AddmmOp>(loc, resultType, bias, input, weight, alpha, beta);
        Value new_res = addmmOp.getResult();
        if(input_shape.size() == 3){
            if(hasStaticShape(inputType.getSizes())){
                ArrayRef<int64_t> viewInput_size = result_shape;
                RankedTensorType DenseInputType = RankedTensorType::get({3}, rewriter.getIntegerType(64));
                auto denseAttr = DenseElementsAttr::get(DenseInputType, viewInput_size);

                auto vewInputType = diffusion_graph::TupleType::get(context, rewriter.getIntegerType(64));
                auto viewInputTupleOp = rewriter.create<diffusion_graph::ConstTupleOp>(unknownLoc, vewInputType, denseAttr);

                auto viewResultType = diffusion_graph::ValueTensorType::get(context, viewInput_size, inputType.getDtype());
                new_res = rewriter.create<diffusion_graph::ViewOp>(loc, viewResultType, new_res, viewInputTupleOp);
            } else {
                size3 = rewriter.create<diffusion_graph::SizeOp>(loc, intType, weight, dim1);
                Value value_array[] = {size0, size1, size3}; 
                ArrayRef<Value> Operand_array(value_array, 3);
                ValueRange OperandList(Operand_array);

                Type viewListResultType = diffusion_graph::ListType::get(context, intType);
                auto viewInputListOp = rewriter.create<diffusion_graph::ListOp>(loc, viewListResultType, OperandList);
                
                // int64_t new_result_view_size_c[] = {DYNAMIC_SIZE, input_shape[2]};
                // ArrayRef<int64_t> viewResultSize(new_result_view_size_c, 2);
                auto vllm_resultType = cast<diffusion_graph::ValueTensorType>(resultType);
                auto viewResultType = diffusion_graph::ValueTensorType::get(context, vllm_resultType.getSizes(), vllm_resultType.getDtype());
                
                new_res = rewriter.create<diffusion_graph::ViewOp>(loc, viewResultType, new_res, viewInputListOp);
            }
            
        }

        addRes.replaceAllUsesWith(new_res);

        rewriter.eraseOp(user_op);
        rewriter.eraseOp(op);
        
    }

    return success();
}

template<> 
LogicalResult RecomposeSimpleOps<diffusion_graph::BroadCastIndexOp>::matchAndRewrite(diffusion_graph::BroadCastIndexOp op, PatternRewriter &rewriter) const{
    

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    SmallVector<Operation*> patternOpsToRecompose;
    vector<int64_t> resizeDims;
    if(!isPatternUpsampleNearestFunc(op, patternOpsToRecompose, resizeDims))
        return failure();


    // Lower to arith.constant with StringAttr - output is generic string
    llvm::StringRef str("nearest");
    auto stringAttr = rewriter.getStringAttr(str);
    auto constantStringOp = rewriter.create<diffusion_graph::ConstantStringOp>(loc, stringAttr);

    Value Input = op.getOperand(0);
    Value result = op.getResult();
    Type resultType = result.getType();
    Type InputType = Input.getType();

    auto inputVllmType = cast<diffusion_graph::ValueTensorType>(InputType);
    auto resultVllmType = cast<diffusion_graph::ValueTensorType>(resultType);

    auto inputShape = inputVllmType.getSizes();
    auto resultShape = resultVllmType.getSizes();

    float scale_factor = (float)resultShape[2] / (float)inputShape[2];

    auto ScaleFactorConstOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getFloatAttr(rewriter.getF32Type(), scale_factor));
    
    rewriter.replaceOpWithNewOp<diffusion_graph::UpsampleOp>(op, resultType, Input, ScaleFactorConstOp, constantStringOp);

    for ( Operation* op : patternOpsToRecompose){
        if(!op->hasSuccessors())
            rewriter.eraseOp(op);
    }

    return success();

    
}

template<> 
LogicalResult RecomposeSimpleOps<diffusion_graph::LayerNormOp>::matchAndRewrite(diffusion_graph::LayerNormOp op, PatternRewriter &rewriter) const{
    Operation* BroadCastOp = nullptr;
    for(auto* user_op : op->getUsers()){
        if(auto user_op_cast = dyn_cast<diffusion_graph::BroadCastIndexOp>(user_op)){
            BroadCastOp = user_op;
            break;
        }
    }
    if(!BroadCastOp){
        return failure();
    }

    Value result = BroadCastOp->getResult(0);
    Type resultType = result.getType();

    Value listOperand = BroadCastOp->getOperand(1);
    Value input = BroadCastOp->getOperand(0);
    Operation* listOp = listOperand.getDefiningOp<diffusion_graph::ListOp>();
    Value indices = listOp->getOperand(1);

    Location loc = BroadCastOp->getLoc();
    // Move the insertion point to just before BroadCastOp so that 'input'
    // (the LayerNormOp result) and 'indices' dominate the newly created ops.
    rewriter.setInsertionPoint(BroadCastOp);
    Value dim = rewriter.create<arith::ConstantIntOp>(loc, -1, rewriter.getIntegerType(32));
    Value indexSelectResult = rewriter.create<diffusion_graph::IndexSelectOp>(loc, resultType, input, dim, indices);

    result.replaceAllUsesWith(indexSelectResult);
    
    rewriter.eraseOp(BroadCastOp);
    rewriter.eraseOp(listOp);
    
    return success();
}


template<> 
LogicalResult RecomposeSimpleOps<diffusion_graph::BMMOp>::matchAndRewrite(diffusion_graph::BMMOp op, PatternRewriter &rewriter) const{
    
    Value input = op.getOperand(0);
    Value value = op.getOperand(1);

    Value result = op.getResult();
    Type resultType = result.getType();

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    auto softmax_op = input.getDefiningOp<diffusion_graph::SoftmaxOp>();
    if(!softmax_op){
        return failure();
    }

    input = softmax_op.getOperand(0);


    auto bmm2_op = input.getDefiningOp<diffusion_graph::BMMOp>();
    if(!bmm2_op){
        return failure();
    }


    Value query = bmm2_op.getOperand(0);
    Value key = bmm2_op.getOperand(1);

    auto mul1_op = query.getDefiningOp<diffusion_graph::MulOp>();
    if(!mul1_op){
        return failure();
    }

    query = mul1_op.getOperand(0);

    auto mul2_op = key.getDefiningOp<diffusion_graph::MulOp>();
    if(!mul2_op){
        return failure();
    }


    key = mul2_op.getOperand(0);

    auto transpose_op = key.getDefiningOp<diffusion_graph::TransposeOp>();
    if(!transpose_op){
        return failure();
    }


    key = transpose_op.getOperand(0);

    Value falseOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(0));
    
    Type NoneType = diffusion_graph::NoneType::get(context);
    Type f32Type = rewriter.getF32Type();
    

    Value NoneOp = rewriter.create<diffusion_graph::ConstantNoneOp>(loc, NoneType);
    Value dropout = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getFloatAttr(f32Type, 0.0f));

    Value newResult = rewriter.replaceOpWithNewOp<diffusion_graph::ScaledDotProductAttentionOp>(op, resultType, query, key, value, NoneOp, dropout, falseOp, NoneOp, falseOp);

    rewriter.eraseOp(softmax_op);
    rewriter.eraseOp(bmm2_op);
    rewriter.eraseOp(mul2_op);
    rewriter.eraseOp(mul1_op);
    rewriter.eraseOp(transpose_op);

    return success();
    
}


template<>
LogicalResult RecomposeSimpleOps<diffusion_graph::TransposeOp>::matchAndRewrite(diffusion_graph::TransposeOp op, PatternRewriter &rewriter) const{

    SmallVector<Operation*> patternOpsToDrop;
    SmallVector<Value> inputValues;

    Type resultType = op.getResult().getType();
    if(!isPatternSDPA(op, patternOpsToDrop, inputValues)){
        return failure();
    }

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value query = inputValues[0];
    Value key = inputValues[1];
    Value value = inputValues[2];

    Value falseOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getBoolAttr(0));
    
    Type NoneType = diffusion_graph::NoneType::get(context);
    Type f32Type = rewriter.getF32Type();
    

    Value NoneOp = rewriter.create<diffusion_graph::ConstantNoneOp>(loc, NoneType);
    Value dropout = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getFloatAttr(f32Type, 0.0f));

    Value newResult = rewriter.replaceOpWithNewOp<diffusion_graph::ScaledDotProductAttentionOp>(op, resultType, query, key, value, NoneOp, dropout, falseOp, NoneOp, falseOp);

    for(auto* DropOp : patternOpsToDrop){
        if(!DropOp->hasSuccessors())
            rewriter.eraseOp(DropOp);
    }
    
    return success();
}
} //namespace


namespace{
class RecomposeSimpleOpsToComplex : public RecomposeSimpleOpsToComplexPassBase<RecomposeSimpleOpsToComplex> {
public:
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<diffusion_graph::DiffusionGraphIRDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() override{
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        target.addLegalDialect<diffusion_graph::DiffusionGraphIRDialect, arith::ArithDialect, func::FuncDialect>();

        RewritePatternSet patterns(context);

        patterns.add<RecomposeSimpleOps<diffusion_graph::MatmulOp>>(context);
        patterns.add<RecomposeSimpleOps<diffusion_graph::BroadCastIndexOp>>(context);
        patterns.add<RecomposeSimpleOps<diffusion_graph::LayerNormOp>>(context);
        patterns.add<RecomposeSimpleOps<diffusion_graph::BMMOp>>(context);
        patterns.add<RecomposeSimpleOps<diffusion_graph::TransposeOp>>(context);

        GreedyRewriteConfig config;
        config.useTopDownTraversal = true;
        config.maxIterations = GreedyRewriteConfig::kNoLimit;

        // llvm::outs() << getOperation() << "\n";
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                            config))) {
            return signalPassFailure();
        }

    }
};
} //namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::diffusion_graph::createRecomposeSimpleOpsToComplexOps(){
    return std::make_unique<RecomposeSimpleOpsToComplex>();
}


