#include "vllm_graph/Conversion/Passes.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::vllm_graph;

void mlir::vllm_graph::registervLLMGraphPasses(){
    ::registerPasses();
}

void createTorchTovLLMGraphPipeline(OpPassManager &pm){

    pm.addPass(createConvertGlobalFunctionPass());
}