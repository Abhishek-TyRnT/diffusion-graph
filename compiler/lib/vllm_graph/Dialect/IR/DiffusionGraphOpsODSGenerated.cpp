#include "vllm_graph/Dialect/IR/DiffusionGraphOps.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::diffusion_graph;

#define GET_OP_CLASSES
#include "vllm_graph/Dialect/IR/DiffusionGraphOps.cpp.inc"

ParseResult diffusion_graph::parseDefaultDiffusionGraphOp(OpAsmParser &parser,
                                       OperationState &result, int numOperands,
                                       int numResults) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/numOperands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  if (numOperands > 0) {
    SmallVector<Type> operandTypes;
    if (parser.parseTypeList(operandTypes))
      return failure();
    if (parser.resolveOperands(operands, operandTypes, loc, result.operands))
      return failure();
  }
  if (numOperands > 0 && numResults > 0) {
    if (parser.parseArrow())
      return failure();
  }
  if (numResults > 0) {
    if (parser.parseTypeList(result.types))
      return failure();
  }
  return success();
}

void diffusion_graph::printDefaultDiffusionGraphOp(OpAsmPrinter &p, Operation *op, int numOperands,
                                int numResults) {
  if (numOperands > 0) {
    p << ' ';
    llvm::interleaveComma(op->getOperands(), p);
  }
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{});
  p << " : ";
  if (numOperands > 0)
    llvm::interleaveComma(op->getOperandTypes(), p);
  if (numOperands > 0 && numResults > 0)
    p << " -> ";
  if (numResults > 0)
    llvm::interleaveComma(op->getResultTypes(), p);
}
