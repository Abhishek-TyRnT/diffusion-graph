#ifndef DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSES_H_
#define DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace diffusion_graph {

#include "diffusion_graph/Dialect/Transform/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertGlobalFunctionPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineDialectResourcesDictPass();

std::unique_ptr<OperationPass<func::FuncOp>> createRecomposeSimpleOpsToComplexOps();

std::unique_ptr<OperationPass<func::FuncOp>> createStaticOpMaterializationPass();

std::unique_ptr<OperationPass<ModuleOp>> createDiffusionGraphFunctionPartitionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createContiguousInsertionPass();

void createTorchToDiffusionGraphPipeline(PassManager &pm);

void registerDiffusionGraphPasses();

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "diffusion_graph/Dialect/Transform/Passes.h.inc"

} //diffusion_graph
} //mlir

#endif //DIFFUSION_GRAPH_DIALECT_TRANSFORM_PASSES_H_

