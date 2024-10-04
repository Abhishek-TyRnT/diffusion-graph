
#ifndef VLLM_GRAPH_CONVERSION_PASSDETAIL_H
#define VLLM_GRAPH_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace vllm_graph {

#define GEN_PASS_CLASSES
#include "vllm_graph/Conversion/Passes.h.inc"

} // namespace vllm_graph
} // end namespace mlir

#endif //VLLM_GRAPH_CONVERSION_PASSDETAIL_H