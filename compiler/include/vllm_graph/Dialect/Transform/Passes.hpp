#ifndef VLLM_GRAPH_DIALECT_TRANSFORM_PASSES_H_
#define VLLM_GRAPH_DIALECT_TRANSFORM_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace vllm_graph {

#include "vllm_graph/Dialect/Transform/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertGlobalFunctionPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineDialectResourcesDictPass();

std::unique_ptr<OperationPass<func::FuncOp>> createRecomposeSimpleOpsToComplexOps();

void createTorchTovLLMGraphPipeline(PassManager &pm);

void registervLLMGraphPasses();

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "vllm_graph/Dialect/Transform/Passes.h.inc"

} //vllm_graph
} //mlir

#endif //VLLM_GRAPH_DIALECT_TRANSFORM_PASSES_H_

