
#include "InitAll.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "vllm_graph/Conversion/Passes.hpp"
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include <iostream>

void mlir::vllm_graph::registerAllDialects(mlir::DialectRegistry &registry)
{
    //registry.insert<mlir::func::FuncDialect>();
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
    registry.insert<mlir::vllm_graph::vLLMGraphIRDialect>();
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
    registry.insert<mlir::torch::Torch::TorchDialect>();
    std::cout << __FILE__ << " " << __LINE__ << std::endl;

}

void mlir::vllm_graph::registerAllPasses() {
    mlir::vllm_graph::registervLLMGraphPasses();
    mlir::vllm_graph::registerConversionPasses();
}

