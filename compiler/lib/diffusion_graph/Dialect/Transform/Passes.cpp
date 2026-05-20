#include "diffusion_graph/Conversion/Passes.hpp"
#include "diffusion_graph/Dialect/Transform/Passes.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::diffusion_graph;

void mlir::diffusion_graph::registerDiffusionGraphPasses(){
    ::registerPasses();
}

void mlir::diffusion_graph::createTorchToDiffusionGraphPipeline(PassManager &pm){

    pm.addPass(createConvertGlobalFunctionPass());
    mlir::OpPassManager &FuncOpPM = pm.nest<mlir::func::FuncOp>();
    FuncOpPM.addPass(createTorchToDiffusionGraph());

    // pm.addPass(createCanonicalizerPass());
    FuncOpPM.addPass(createCanonicalizerPass());

    FuncOpPM.addPass(createRecomposeSimpleOpsToComplexOps());
    FuncOpPM.addPass(createStaticOpMaterializationPass());
    FuncOpPM.addPass(createContiguousInsertionPass());
    pm.addPass(createDiffusionGraphFunctionPartitionPass());
    pm.addPass(createCanonicalizerPass());

}