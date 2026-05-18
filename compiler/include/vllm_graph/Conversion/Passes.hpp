
#ifndef VLLM_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H
#define VLLM_GRAPH_CONVERSION_TORCH_TO_DIFFUSIONGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace diffusion_graph{

void registerConversionPasses();
} // diffusion_graph
} // mlir

namespace mlir::diffusion_graph {

std::unique_ptr<OperationPass<func::FuncOp>> createTorchToDiffusionGraph();

void registerTorchToDiffusionGraphPasses();

} // mlir::diffusion_graph


#endif
