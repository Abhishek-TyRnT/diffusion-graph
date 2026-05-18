#ifndef DIFFUSION_GRAPH_UTILS_UTILS_H_
#define DIFFUSION_GRAPH_UTILS_UTILS_H_

#include "vllm_graph/Conversion/TorchTovLLMGraph.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Types.h"

namespace mlir{
namespace diffusion_graph{

Type convertTorchvTypeToDGvType(Type type, MLIRContext *context);


RankedTensorType convertTorchvTypeToTensorType(Type type);

Type convertDGContainedType(Type type, 
                        MLIRContext *context);

}// namespace diffusion_graph

}// namespace mlir
#endif