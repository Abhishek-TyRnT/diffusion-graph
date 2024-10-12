
#ifndef VLLM_GRAPH_CONVERSION_TORCH_TO_VLLMGRAPH_H
#define VLLM_GRAPH_CONVERSION_TORCH_TO_VLLMGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace vllm_graph{

void registerConversionPasses();
} // vllm_graph
} // mlir

namespace mlir::vllm_graph {

std::unique_ptr<OperationPass<func::FuncOp>> createTorchTovLLMGraph();

void registerTorchTovLLMGraphPasses();

} // mlir::vllm_graph


#endif
