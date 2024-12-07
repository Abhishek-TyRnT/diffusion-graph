#include "Interface.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "InitAll.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

using namespace mlir;
using namespace mlir::vllm_graph;


vLLMGraphBase::vLLMGraphBase(){
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllPasses();
    context = new MLIRContext(registry);
    passmanager = std::make_unique<PassManager>(OperationName("builtin.module", context));
    // OpPassManager &Opm = passmanager->nest<mlir::ModuleOp>();
    createTorchTovLLMGraphPipeline(*passmanager);
}

vLLMGraphBase::~vLLMGraphBase(){
    delete context;
}

OwningOpRef<mlir::ModuleOp> vLLMGraphBase::parse(std::string IR){
    // llvm::SourceMgr sourceMgr;
    // // auto fileOrErr = llvm::MemoryBuffer::getFile(IRFile);
    // // if (!fileOrErr) {
    // //     llvm::errs() << "Could not open input file\n";
    // //     return nullptr;
    // // }
    // sourceMgr.AddNewSourceBuffer(IR, llvm::SMLoc());

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
    
    auto fileOrErr = llvm::MemoryBuffer::getFile(IRFile);
    if (!fileOrErr) {
        llvm::errs() << "Could not open input file\n";
        return nullptr;
    }
    //TODO fix this
    return parse("empty string");
}

void vLLMGraphBase::convert(OwningOpRef<ModuleOp> &module){

    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
    }
    
    if(failed(passmanager->run(*module)))
        llvm::errs() << "The Pass failed to run"<< "\n";
}




