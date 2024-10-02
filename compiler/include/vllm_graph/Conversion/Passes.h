
#ifndef VLLM_GRAPH_CONVERSION_TORCH_TO_VLLMGRAPH_H
#define VLLM_GRAPH_CONVERSION_TORCH_TO_VLLMGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::vllm_graph::torch {

std::unique_ptr<OperationPass<func::FuncOp>> createTorchTovLLMGraph();

void registerTorchTovLLMGraphPasses();

} // mlir::vllm_graph::torch

#endif
