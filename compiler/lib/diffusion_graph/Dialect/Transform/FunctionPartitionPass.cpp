
#include "PassDetail.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "diffusion_graph/Dialect/IR/DiffusionGraphDialect.hpp"
#include "diffusion_graph/Dialect/Transform/Passes.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphTypes.hpp"
#include "diffusion_graph/Dialect/IR/DiffusionGraphOps.hpp"
#include "diffusion_graph/Dialect/Patterns/PatternInterpreter.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <unistd.h>
#include <string>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>

using namespace mlir;
using namespace mlir::diffusion_graph;

std::string getRootPath() {
    Dl_info info;
    // Pass a symbol from your own .so
    dladdr((void*)&getRootPath, &info);
    
    std::string fname(info.dli_fname);
    if(fname != "diffusion-graph" && fname != "diffusion-graph-opt"){
        auto so_dir = std::filesystem::canonical(fname).parent_path();
        if(so_dir.filename().string() == "python"){
            so_dir = so_dir.parent_path();
        }
        return so_dir.string();
    } else {
        char buffer[1024];
        ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
        if (len != -1) {
            buffer[len] = '\0';
            std::filesystem::path exePath(buffer);
            return exePath.parent_path().parent_path().string();
        }
    }

    return "";

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
class diffusionGraphFunctionPartitioningPass : public diffusionGraphFunctionPartitioningPassBase<diffusionGraphFunctionPartitioningPass> {
public:

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect, 
                        arith::ArithDialect, 
                        pdl::PDLDialect, 
                        pdl_interp::PDLInterpDialect,
                        diffusion_graph::DiffusionGraphIRDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        // Convert PDL to PDLInterp
        RewritePatternSet patterns(context);
        patterns.add<PoolingLayerSplit>(context, getRootPath() + "/Patterns/poolingLayer.pdl_interp.mlir");
        patterns.add<PoolingLayerSplit>(context, getRootPath() + "/Patterns/CLIPPoolingLayer.pdl_interp.mlir");
    
    // Create a rewriter
        PatternRewriter rewriter(&getContext());
        
        //TODO: Refine this logic.It looks little bit shaky
        for(auto &pattern : patterns.getNativePatterns()){
            for (auto &op : module.getBody()->getOperations()) {
                if(auto funcOp = dyn_cast<func::FuncOp>(op))
                {
                    if(failed(pattern->matchAndRewrite(&op, rewriter))){
                        return signalPassFailure();
                    }
                }
            }
        }
    }
};
} //namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::diffusion_graph::createDiffusionGraphFunctionPartitionPass(){
    return std::make_unique<diffusionGraphFunctionPartitioningPass>();
}
