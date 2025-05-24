

#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

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

