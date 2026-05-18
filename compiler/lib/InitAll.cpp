
#include "InitAll.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "vllm_graph/Conversion/Passes.hpp"
#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/Passes.h"

void mlir::diffusion_graph::registerAllDialects(mlir::DialectRegistry &registry)
{
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::diffusion_graph::vLLMGraphIRDialect>();
    registry.insert<mlir::torch::Torch::TorchDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<mlir::BuiltinDialect>();
}

void mlir::diffusion_graph::registerAllPasses() {
    mlir::diffusion_graph::registervLLMGraphPasses();
    mlir::diffusion_graph::registerConversionPasses();
    mlir::registerTransformsPasses();
}

