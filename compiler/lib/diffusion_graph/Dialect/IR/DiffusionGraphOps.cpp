

#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphTypes.hpp"
#include "diffusion_graph/Utils/Utils.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"


//===----------------------------------------------------------------------===//
// ConstantDeviceOp
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::diffusion_graph;


void ConstantDeviceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getValue());
}

OpFoldResult ValueTensorLiteralOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

OpFoldResult ConstantNoneOp::fold(FoldAdaptor adaptor) {
  return TypeAttr::get(diffusion_graph::NoneType::get(getContext()));
}

OpFoldResult ConstantDeviceOp::fold(FoldAdaptor adaptor){
    return getValueAttr();
}

OpFoldResult ConstantStringOp::fold(FoldAdaptor adaptor){
    return getValueAttr();
}

OpFoldResult SizeOp::fold(FoldAdaptor adaptor){

    ArrayRef<Attribute> operands = adaptor.getOperands();

    if(!operands[1])
      return {};
    
    auto input = getSelf();
    int64_t dim;
    if (auto intAttr = dyn_cast<IntegerAttr>(operands[1])) {
      dim = intAttr.getInt();
    } else {
      return {};
    }
    
    
    auto type = cast<diffusion_graph::ValueTensorType>(input.getType());
    ArrayRef<int64_t> sizes = type.getSizes();

    if(sizes[dim] == DYNAMIC_SIZE)
      return {};
    
    Type resultType = getResult().getType();

    return IntegerAttr::get(resultType, sizes[dim]);
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor){

    // return {};
    ArrayRef<Attribute> operands = adaptor.getOperands();

    if(!operands[0] || !operands[1])
      return {};
    if(getOperand(0).getType() != getOperand(1).getType())
      return {};
    Type resultType = getResult().getType();
    if (auto intAttr = dyn_cast<IntegerAttr>(operands[1])) {
      int64_t constant_1 = intAttr.getInt();
      int64_t constant_2 = dyn_cast<IntegerAttr>(operands[0]).getInt();

      int64_t prod = constant_1 * constant_2;
      return IntegerAttr::get(resultType, prod);


    } else if(auto floatAttr = dyn_cast<FloatAttr>(operands[1])){
      llvm::APFloat constant_1 = floatAttr.getValue();
      llvm::APFloat constant_2 = dyn_cast<FloatAttr>(operands[0]).getValue();

      llvm::APFloat prod = constant_1 * constant_2;
      return FloatAttr::get(resultType, prod);

    } else {
      return {};
    }
    
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
    struct EliminateCast : public OpRewritePattern<CastOp> {
        using OpRewritePattern<CastOp>::OpRewritePattern;
        
        LogicalResult matchAndRewrite(CastOp op,
                                      PatternRewriter &rewriter) const override {
            Value result = op.getResult();
            Value input = op.getOperand();
            result.replaceAllUsesWith(input);
            rewriter.eraseOp(op);
            return success();
        }
    };  
    
    patterns.add<EliminateCast>(context);
}

void SizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
    struct EliminateSizeOp : public OpRewritePattern<SizeOp> {
        using OpRewritePattern<SizeOp>::OpRewritePattern;
        
        LogicalResult matchAndRewrite(SizeOp op,
                                      PatternRewriter &rewriter) const override {
      // Check if the operation result has any uses
          if (op.getResult().use_empty()) {
            // No uses found, eliminate the operation
            rewriter.eraseOp(op);
            return success();
          }
          // Operation has uses, don't eliminate
          return failure();
            }
    };    
    
    patterns.add<EliminateSizeOp>(context);
}

void DtypeCastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  // Pattern to eliminate BroadCastOp if its result has no uses
  struct EliminateUnusedDtypeCast : public OpRewritePattern<DtypeCastOp> {
    using OpRewritePattern<DtypeCastOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(DtypeCastOp op,
                                  PatternRewriter &rewriter) const override {
      // Check if the operation result has any uses
      if (op.getResult().use_empty()) {
        // No uses found, eliminate the operation
        rewriter.eraseOp(op);
        return success();
      }
      // Operation has uses, don't eliminate
      return failure();
    }
  };  
  
  patterns.add<EliminateUnusedDtypeCast>(context);
}


void BroadCastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  // Pattern to eliminate BroadCastOp if its result has no uses
  //TODO: Make it common struct to be used for all canocialization pattern with no uses
  struct EliminateUnusedBroadCast : public OpRewritePattern<BroadCastOp> {
    using OpRewritePattern<BroadCastOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(BroadCastOp op,
                                  PatternRewriter &rewriter) const override {
      // Check if the operation result has any uses
      if (op.getResult().use_empty()) {
        // No uses found, eliminate the operation
        rewriter.eraseOp(op);
        return success();
      }
      // Operation has uses, don't eliminate
      return failure();
    }
  };  
  
  patterns.add<EliminateUnusedBroadCast>(context);
}

void ViewOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  // Pattern to eliminate ViewOp if its result has no uses
  struct EliminateRedundantView : public OpRewritePattern<ViewOp> {
    using OpRewritePattern<ViewOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(ViewOp op,
                                  PatternRewriter &rewriter) const override {
      
        Value input = op.getOperand(0);
        Value result = op.getResult();

        if(input.getType() == result.getType()){
            result.replaceAllUsesWith(input);
            rewriter.eraseOp(op);
            return success();
        }

        return failure();
        
    }
  };  
  
  patterns.add<EliminateRedundantView>(context);
}

// static Type promoteDtype(Type a, Type b) {
//   if (a == b) return a;

//   if (isa<arith::IntType>(a) && isa<arith::IntType>(b)){
//     auto int_a = cast<arith::IntType>(a);
//     auto int_b = cast<arith::IntType>(b);
//     if (int_a.getWidth() > int_b.getWidth()){
//       return a;
//     } else {
//       return b;
//     }
//   }
//   if (isa<arith::FloatType>(a) && isa<arith::FloatType>(b)){
//     auto float_a = cast<arith::FloatType>(a);
//     auto float_b = cast<arith::FloatType>(b);
//     if (float_a.getWidth() > float_b.getWidth()){
//       return a;
//     } else {
//       return b;
//     }
//   }

//   return Type();
// }

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  auto lhsType = dyn_cast<diffusion_graph::ValueTensorType>(operands[0].getType());
  auto rhsType = dyn_cast<diffusion_graph::ValueTensorType>(operands[1].getType());
  if (!lhsType && !rhsType)
    return failure();

  // dtype promotion rule: e.g. fp16 + fp32 -> fp32, int + float -> float
  Type resultElemType = promoteDtype(lhsType.getOptionalDtype(),
                                      rhsType.getOptionalDtype());
  if (!resultElemType)
    return emitOptionalError(location, "unsupported dtype combination for add: ",
                              lhsType.getOptionalDtype(), " and ",
                              rhsType.getOptionalDtype());

  // shape rule: broadcasted shape (or just require equal shapes if your
  // dialect doesn't support broadcasting)

  inferredReturnTypes.push_back(
      diffusion_graph::ValueTensorType::get(context, lhsType.getOptionalSizes(), resultElemType));
  return success();
}


