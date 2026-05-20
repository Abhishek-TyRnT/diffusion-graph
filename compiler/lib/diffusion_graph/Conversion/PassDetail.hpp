
#ifndef DIFFUSION_GRAPH_CONVERSION_PASSDETAIL_H
#define DIFFUSION_GRAPH_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace diffusion_graph {

#define GEN_PASS_CLASSES
#include "diffusion_graph/Conversion/Passes.h.inc"

} // namespace diffusion_graph
} // end namespace mlir

#endif //DIFFUSION_GRAPH_CONVERSION_PASSDETAIL_H