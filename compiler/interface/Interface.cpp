#include "Interface.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "InitAll.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::vllm_graph;

vLLMGraph::vLLMGraph(){
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllPasses();
    context = new MLIRContext(registry);
    passmanager = std::make_unique<PassManager>(OperationName("builtin.module", context));
    // OpPassManager &Opm = passmanager->nest<mlir::ModuleOp>();
    createTorchTovLLMGraphPipeline(*passmanager);
}

vLLMGraph::~vLLMGraph(){
    delete context;
}

OwningOpRef<mlir::ModuleOp> vLLMGraph::convert(std::string IRFile){
    llvm::SourceMgr sourceMgr;
    auto fileOrErr = llvm::MemoryBuffer::getFile(IRFile);
    if (!fileOrErr) {
        llvm::errs() << "Could not open input file\n";
        return nullptr;
    }
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    // Parse the file into an MLIR module.
    mlir::ParserConfig parserConfig(context);
    OwningOpRef<ModuleOp> module =
        mlir::parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
        return nullptr;
    }
    
    if(failed(passmanager->run(*module)))
        llvm::errs() << "The Pass failed to run"<< "\n";
    llvm::outs() << *module << "\n";
    return module;
}




