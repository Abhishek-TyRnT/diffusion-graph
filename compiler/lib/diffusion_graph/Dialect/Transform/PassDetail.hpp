#ifndef DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_
#define DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
namespace diffusion_graph {

#define GEN_PASS_CLASSES
#include "diffusion_graph/Dialect/Transform/Passes.h.inc"

} // namespace diffusion_graph
} // end namespace mlir

#endif // DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSDETAIL_H_
