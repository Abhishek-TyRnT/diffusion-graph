#include "Interface.hpp"

using namespace mlir;

int main(int argc, char **argv) {

    if(argv[1] == ""){
        llvm::errs() << "Input File not provided" << "\n";
        exit(-1);
    }
    mlir::vllm_graph::vLLMGraphBase graph;
    OwningOpRef<mlir::ModuleOp> moduleOp = graph.convert(argv[1]);
    llvm::outs() << *moduleOp << "\n";
    return 0;

}