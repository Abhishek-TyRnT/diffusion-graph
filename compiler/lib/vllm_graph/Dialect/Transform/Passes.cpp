#include "vllm_graph/Conversion/Passes.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vllm_graph;

void mlir::vllm_graph::registervLLMGraphPasses(){
    ::registerPasses();
}

void mlir::vllm_graph::createTorchTovLLMGraphPipeline(PassManager &pm){

    pm.addPass(createInlineDialectResourcesDictPass());
    pm.addPass(createConvertGlobalFunctionPass());
    mlir::OpPassManager &FuncOpPM = pm.nest<mlir::func::FuncOp>();
    FuncOpPM.addPass(createTorchTovLLMGraph());
    FuncOpPM.addPass(createvLLMCanonicalizerPass());
    FuncOpPM.addPass(createvLLMCanonicalizerPass());
    FuncOpPM.addPass(createRecomposeSimpleOpsToComplexOps());
    pm.addPass(createvLLMFunctionPartitionPass());
    FuncOpPM.addPass(createvLLMCanonicalizerPass());
}