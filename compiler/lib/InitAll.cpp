
#include "InitAll.hpp"
#include "InitAllc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "vllm_graph/Conversion/Passes.hpp"
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "vllm_graph/Dialect/Transform/Passes.hpp"

using namespace mlir::vllm_graph;


void mlir::vllm_graph::registerAllDialects(mlir::DialectRegistry &registry)
{
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::vllm_graph::vLLMGraphIRDialect>();
    registry.insert<mlir::torch::Torch::TorchDialect>();

}

void mlir::vllm_graph::registerAllPasses() {
    mlir::vllm_graph::registervLLMGraphPasses();
    mlir::vllm_graph::registerConversionPasses();
}

void registervLLMGraphDialect(MlirContext &contextc){
    mlir::MLIRContext *context = reinterpret_cast<mlir::MLIRContext*>(contextc.ptr);
    mlir::DialectRegistry registry = context->getDialectRegistry();
    registerAllDialects(registry);

}

void registervLLMGraphPasses(){
    registerAllPasses();
}

