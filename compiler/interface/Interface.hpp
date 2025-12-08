#ifndef VLLM_GRAPH_INTERFACE_H_
#define VLLM_GRAPH_INTERFACE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>
#include <string>

using namespace mlir;

namespace mlir{
namespace vllm_graph{

class vLLMGraphBase{

public:
    vLLMGraphBase();
    void convert(OwningOpRef<mlir::ModuleOp> &module);
    OwningOpRef<mlir::ModuleOp> parse(std::string IR);
    OwningOpRef<mlir::ModuleOp> parseFromFile(std::string IRFile);
    ~vLLMGraphBase();

protected:
    MLIRContext *context;
    std::unique_ptr<PassManager> passmanager;
    // DenseMap to store dialect resources: key -> resource blob data
    llvm::DenseMap<StringRef, mlir::ArrayRef<char>> dialectResourcesMap;
    
    // Helper method to extract dialect resources from a module
    void extractDialectResources(mlir::ModuleOp module);
};

} //vllm_graph
} //mlir

#endif