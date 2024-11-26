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
    OwningOpRef<mlir::ModuleOp> convert(std::string IRFile);
    //pybind11::pyobject compile(std::string IRFile);
    void print(ModuleOp module);
    ~vLLMGraphBase();

protected:
    MLIRContext *context;
    std::unique_ptr<PassManager> passmanager;
};

} //vllm_graph
} //mlir

#endif