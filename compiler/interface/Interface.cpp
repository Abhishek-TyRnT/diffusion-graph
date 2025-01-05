#include "Interface.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "InitAll.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::vllm_graph;


vLLMGraphBase::vLLMGraphBase(){
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllPasses();
    context = new MLIRContext(registry);
    passmanager = std::make_unique<PassManager>(OperationName("builtin.module", context));
    createTorchTovLLMGraphPipeline(*passmanager);
}

vLLMGraphBase::~vLLMGraphBase(){
    delete context;
}

OwningOpRef<mlir::ModuleOp> vLLMGraphBase::parse(std::string IR){

    // Parse the file into an MLIR module.
    mlir::ParserConfig parserConfig(context);
    auto OpRef =
        mlir::parseSourceString(IR, parserConfig);
    
    if (!OpRef) {
        llvm::errs() << "Failed to parse MLIR IR\n";
    }

    // Cast OwningOpRef<Operation*> to OwningOpRef<ModuleOp>
    if (!isa<ModuleOp>(OpRef.get())) {
        llvm::errs() << "The operation is not a ModuleOp\n";
    }

    auto moduleOpRef = mlir::OwningOpRef<ModuleOp>(mlir::cast<ModuleOp>(OpRef.release()));

    return moduleOpRef;
}

OwningOpRef<mlir::ModuleOp> vLLMGraphBase::parseFromFile(std::string IRFile){
    
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
    return module;
}

void vLLMGraphBase::convert(OwningOpRef<ModuleOp> &module){

    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
        std::exit(-1);
    }
    
    if(failed(passmanager->run(*module))){
        llvm::errs() << *module << "\n";   
        llvm::errs() << "The Pass failed to run"<< "\n";
        std::exit(-1);
    }
}




