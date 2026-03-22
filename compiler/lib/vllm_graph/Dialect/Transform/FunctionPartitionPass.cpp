
#include "PassDetail.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphDialect.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"
#include "vllm_graph/Dialect/Patterns/PatternInterpreter.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <unistd.h>
#include <string>
#include <dlfcn.h>
#include <filesystem>

using namespace mlir;
using namespace mlir::vllm_graph;

std::string getRootPath() {
    Dl_info info;
    // Pass a symbol from your own .so
    dladdr((void*)&getRootPath, &info);
    
    auto so_dir = std::filesystem::canonical(info.dli_fname).parent_path();
    so_dir = so_dir.parent_path();
    return so_dir.string();

}

class PoolingLayerSplit : public RewritePattern {
    //TODO: Remove the hardcoding and take the input from the user
    // std::string pattern_path = "/home/abhishek/vllm-project/build/Patterns/poolingLayer.pdl_interp.mlir";
    const PDLInterpMatcher pattern_interpreter;

public:
    PoolingLayerSplit(MLIRContext *context, std::string pattern_path)
      : RewritePattern(func::FuncOp::getOperationName(), 1, context), pattern_interpreter(context, pattern_path) {
        ;
        // pattern_interpreter.loadPDLInterpFile(pattern_path);
      }

    bool match(Operation *op) const {

        func::FuncOp func = dyn_cast<func::FuncOp>(op);
        if (!func.getSymName().starts_with("main")){
            return false;
        }

        return pattern_interpreter.matchInTree(op);
    }

    LogicalResult rewrite(PatternRewriter &rewriter) const {
        return pattern_interpreter.rewriteModule(rewriter);
    }

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {

        
        if(match(op)){            
            LogicalResult result = rewrite(rewriter);
            return result;
        }

        return success();

    }
};

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

        // Convert PDL to PDLInterp
        RewritePatternSet patterns(context);
        patterns.add<PoolingLayerSplit>(context);

        // TODO: Iterate through all the patterns
         auto &pattern = *patterns.getNativePatterns().begin();
    
    // Create a rewriter
        PatternRewriter rewriter(&getContext());
        
        //TODO: Refine this logic.It looks little bit shaky
        for (auto &op : module.getBody()->getOperations()) {
            if(auto funcOp = dyn_cast<func::FuncOp>(op))
            {
                if(failed(pattern->matchAndRewrite(&op, rewriter))){
                    return signalPassFailure();
                }

            }
        }

    }
};
} //namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::vllm_graph::createvLLMFunctionPartitionPass(){
    return std::make_unique<vLLMFunctionPartitioningPass>();
}
