#ifndef DIFFUSION_GRAPH_DIALECT_PATTERNS_NATIVEREWRITES_H
#define DIFFUSION_GRAPH_DIALECT_PATTERNS_NATIVEREWRITES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"

namespace mlir {
namespace diffusion_graph{


bool createPoolingFunc(Value rootOpResult, PatternRewriter& rewriter);

bool createCLIPPoolingFunc(Value rootOpResult, PatternRewriter& rewriter);

} //diffusion_graph
} //mlir

#endif