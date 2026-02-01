#include "Interface.hpp"
#include "vllm_graph/Dialect/Transform/Passes.hpp"
#include "InitAll.hpp"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/IR/DialectResourceBlobManager.h"

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

// Helper function to extract dialect resources from the module
void vLLMGraphBase::extractDialectResources(mlir::ModuleOp module) {
    // Clear existing resources
    dialectResourcesMap.clear();
    
    // Walk through all operations in the module to find DenseResourceElementsAttr
    module->walk([this](mlir::Operation *op) {
        // Check all attributes of the operation
        for (auto namedAttr : op->getAttrs()) {
            if (auto resourceAttr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(namedAttr.getValue())) {
                // Get the resource handle and blob
                auto handle = resourceAttr.getRawHandle();
                StringRef key = handle.getKey();
                
                // Get the blob data
                if (auto *blob = handle.getBlob()) {
                    llvm::ArrayRef<char> data = blob->getData();
                    dialectResourcesMap[key] = data;
                }
            }
        }
    });
}

OwningOpRef<mlir::ModuleOp> vLLMGraphBase::parse(std::string IR){
    // Parse the file into an MLIR module with resource metadata support.
    mlir::FallbackAsmResourceMap resourceMap;
    mlir::ParserConfig parserConfig(context, /*verifyAfterParse=*/true, &resourceMap);
    auto OpRef = mlir::parseSourceString(IR, parserConfig);
    
    if (!OpRef) {
        llvm::errs() << "Failed to parse MLIR IR\n";
        return nullptr;
    }

    // Cast OwningOpRef<Operation*> to OwningOpRef<ModuleOp>
    if (!isa<ModuleOp>(OpRef.get())) {
        llvm::errs() << "The operation is not a ModuleOp\n";
        return nullptr;
    }

    auto moduleOpRef = mlir::OwningOpRef<ModuleOp>(mlir::cast<ModuleOp>(OpRef.release()));
    
    // Extract dialect resources from the parsed module
    if (moduleOpRef) {
        extractDialectResources(moduleOpRef.get());
    }

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

    // Parse the file into an MLIR module with resource metadata support.
    mlir::FallbackAsmResourceMap resourceMap;
    mlir::ParserConfig parserConfig(context, /*verifyAfterParse=*/true, &resourceMap);
    
    OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
    
    // Extract dialect resources from the parsed module
    if (module) {
        extractDialectResources(module.get());
    }

    return module;
}

void vLLMGraphBase::convert(OwningOpRef<ModuleOp> &module){

    if (!module) {
        llvm::errs() << "Error parsing MLIR file\n";
        std::exit(-1);
    }
    

    if(failed(passmanager->run(*module))){   
        llvm::errs() << "The Pass failed to run"<< "\n";
        std::exit(-1);
    }
}




