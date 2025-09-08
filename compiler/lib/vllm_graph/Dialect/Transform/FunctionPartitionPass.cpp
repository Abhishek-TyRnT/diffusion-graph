
#include "PassDetail.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include <unistd.h>
#include <string>

using namespace mlir;




std::string getRootPath() {
    char buffer[1024];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        std::filesystem::path exePath(buffer);
        return exePath.parent_path().parent_path().string();
    }
    return "";

}

namespace {
class vLLMFunctionPartitioningPass : public vLLMFunctionPartitioningPassBase<vLLMFunctionPartitioningPass> {
public:

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, 
                        arith::ArithDialect, 
                        pdl::PDLDialect, 
                        pdl_interp::PDLInterpDialect,
                        vllm_graph::vLLMGraphIRDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();
        
        // Load PDL pattern from file or embed it
        OwningOpRef<ModuleOp> pdlModule = loadPDLModule(context);
        if (!pdlModule) {
            signalPassFailure();
            return;
        }

        // Convert PDL to PDLInterp
        RewritePatternSet pdlPatterns(context);
        populatePDLToPDLInterpPatterns(pdlPatterns, pdlModule.get());

        // Register native constraints and rewrites
        registerNativeConstraintsAndRewrites(pdlModule.get());

        // Apply patterns using greedy rewriter
        if (failed(applyPatternsAndFoldGreedily(module, std::move(pdlPatterns)))) {
            signalPassFailure();
        }
    }

private:
    OwningOpRef<ModuleOp> loadPDLModule(MLIRContext *context) {
        // Option 1: Load from file
        std::string parent_path = getRootPath();

        //TODO: Hardcoding it, need to figure out later how to make it dynamic 
        std::string pattern_path = parent_path + "/patterns/poolingLayer.pdl"
        return parseSourceFile<ModuleOp>(pattern_path, context);
        
    }

    void registerNativeConstraintsAndRewrites(ModuleOp pdlModule) {
        auto *context = pdlModule.getContext();
        
        // Register native constraints
        PDLPatternModule &patternModule = 
            context->getOrLoadDialect<pdl_interp::PDLInterpDialect>()
                ->getPatternModule();
        
        patternModule.registerConstraintFunction(
            "isConnectedSubgraph", IsConnectedSubgraphConstraint::apply);
        // patternModule.registerConstraintFunction(
        //     "isSplatConstant", IsSplatConstantConstraint::apply);
        
        // Register native rewrites
        patternModule.registerRewriteFunction(
            "getExternalInputs", GetExternalInputsRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "getExternalOutputs", GetExternalOutputsRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "generateUniqueFuncName", GenerateUniqueFuncNameRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "getValueTypes", GetValueTypesRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "createFunctionType", CreateFunctionTypeRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "createFunctionArguments", CreateFunctionArgumentsRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "cloneOperationsToFunction", CloneOperationsToFunctionRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "getClonedOutputs", GetClonedOutputsRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "replaceUsesWithCallResults", ReplaceUsesWithCallResultsRewrite::rewrite);
        patternModule.registerRewriteFunction(
            "eraseOperations", EraseOperationsRewrite::rewrite);
    }

}
} //namespace
