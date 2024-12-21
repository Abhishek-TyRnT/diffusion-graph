#include "Interface.hpp"
#include <cstdlib>

using namespace mlir;

int main(int argc, char **argv) {

    if(argv[1] == ""){
        llvm::errs() << "Input File not provided" << "\n";
        std::exit(-1);
    }
    mlir::vllm_graph::vLLMGraphBase graph;
    OwningOpRef<mlir::ModuleOp> moduleOp = graph.parseFromFile(argv[1]);
    graph.convert(moduleOp);
    llvm::outs() << *moduleOp << "\n";
    return 0;

}