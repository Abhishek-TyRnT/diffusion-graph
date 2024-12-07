#ifndef VLLM_GRAPH_INTERFACE_H_
#define VLLM_GRAPH_INTERFACE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

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
};

} //vllm_graph
} //mlir

#endif