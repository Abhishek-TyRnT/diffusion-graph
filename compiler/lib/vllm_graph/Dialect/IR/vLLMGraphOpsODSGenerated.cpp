#include "vllm_graph/Dialect/IR/vLLMGraphOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vllm_graph;

#define GET_OP_CLASSES
#include "vllm_graph/Dialect/IR/vLLMGraphOps.cpp.inc"