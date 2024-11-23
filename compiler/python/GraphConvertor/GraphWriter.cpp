
#include "GraphWriter.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

void GraphWriter::addOp(mlir::Operation *op){

    for(mlir::Value operand : op->getOperands()){
        if(mlir::isa<mlir::BlockArgument>(operand)){
            std::string argName = "arg" + argCount++;
            mlir::Type argType = operand.getType();
            mlir::vllm_graph::ValueTensorType RankedArg = 
                        mlir::cast<mlir::vllm_graph::ValueTensorType>(argType);
            std::unordered_map<std::string, std::string> map;
            if(RankedArg){
                llvm::ArrayRef<int64_t> shape = RankedArg.getSizes();
                mlir::Type elementType = RankedArg.getDtype();
                std::string elementTypeName;
                llvm::raw_string_ostream os(elementTypeName);
                elementType.print(os); // Prints the element type

                map["vllm_graph_type"] = "vllm_graph.vtensor";
                map["dtype"] = elementTypeName;
                graph[argName] = map;

            }
        }
    }
}
void GraphWriter::build(mlir::OwningOpRef<mlir::ModuleOp> &module){
    module->walk([this](mlir::Operation *op) {
        // Print the operation name
        this->addOp(op);
    });
}