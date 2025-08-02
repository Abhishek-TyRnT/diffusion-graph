

#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
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
using namespace mlir::vllm_graph;


void ConstantDeviceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getValue());
}

OpFoldResult ValueTensorLiteralOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

OpFoldResult ConstantNoneOp::fold(FoldAdaptor adaptor) {
  return TypeAttr::get(vllm_graph::NoneType::get(getContext()));
}

OpFoldResult ConstantDeviceOp::fold(FoldAdaptor adaptor){
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
    
    
    auto type = cast<vllm_graph::ValueTensorType>(input.getType());
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

