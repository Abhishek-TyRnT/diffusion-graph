#ifndef VLLM_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H
#define VLLM_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace diffusion_graph {
std::unique_ptr<OperationPass<func::FuncOp>> createTorchToDiffusionGraph();
}
} // namespace mlir

#endif //VLLM_GRAPH_CONVERSION_TORCH_TO_VLLMGRAPH_H