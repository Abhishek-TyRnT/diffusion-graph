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

    pm.addPass(createConvertGlobalFunctionPass());
    mlir::OpPassManager &FuncOpPM = pm.nest<mlir::func::FuncOp>();
    FuncOpPM.addPass(createTorchTovLLMGraph());

    pm.addPass(createCanonicalizerPass());
    FuncOpPM.addPass(createRecomposeSimpleOpsToComplexOps());
    FuncOpPM.addPass(createStaticOpMaterializationPass());

    pm.addPass(createvLLMFunctionPartitionPass());
    pm.addPass(createCanonicalizerPass());

}