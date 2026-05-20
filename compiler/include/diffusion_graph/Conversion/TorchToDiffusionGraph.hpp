#ifndef DIFFUSION_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H
#define DIFFUSION_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace diffusion_graph {
std::unique_ptr<OperationPass<func::FuncOp>> createTorchToDiffusionGraph();
} // namespace diffusion_graph
} // namespace mlir

#endif //DIFFUSION_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H