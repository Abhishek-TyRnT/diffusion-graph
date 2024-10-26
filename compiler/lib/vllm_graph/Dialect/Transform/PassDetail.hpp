#ifndef VLLM_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_
#define VLLM_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace vllm_graph {

#define GEN_PASS_CLASSES
#include "vllm_graph/Dialect/Transform/Passes.h.inc"

} // namespace vllm_graph
} // end namespace mlir

#endif // VLLM_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_
