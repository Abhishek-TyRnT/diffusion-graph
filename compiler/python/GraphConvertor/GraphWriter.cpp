
#include "GraphWriter.hpp"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include <iostream>

void GraphWriter::addOp(mlir::Operation *op){

    // if(auto FuncOp = mlir::cast<mlir::func::FuncOp>(*op)){

    // }
    for(mlir::Value operand : op->getOperands()){
        if(mlir::isa<mlir::BlockArgument>(operand)){
            std::stringstream argName;
            argName << "arg" << argCount++;
            opMap[operand] = argName.str();
            std::get<std::vector<std::string>>(graph["entrypoint"]).push_back(argName.str());
            mlir::Type argType = operand.getType();
            mlir::vllm_graph::ValueTensorType RankedArg = 
                        mlir::cast<mlir::vllm_graph::ValueTensorType>(argType);
            std::unordered_map<std::string, NestedValueType> map;
            if(RankedArg){
                llvm::ArrayRef<int64_t> shape = RankedArg.getSizes();
                std::vector<int64_t> shapeVec(shape.begin(), shape.end());
                mlir::Type elementType = RankedArg.getDtype();
                std::string elementTypeName;
                llvm::raw_string_ostream os(elementTypeName);
                elementType.print(os); // Prints the element type

                map["vllm_graph_type"] = "vllm_graph.vtensor";
                map["dtype"] = elementTypeName;
                map["shape"] = shapeVec;

            }

            graph[argName.str()] = map;
        }
    }

    uint resCount = 0;
    for(mlir::Value res : op->getResults()){

        std::stringstream ssa_id;
        ssa_id << opCount << "." << resCount++;
        std::unordered_map<std::string, NestedValueType> map;
        mlir::Type resType = res.getType();
        mlir::vllm_graph::ValueTensorType Rankedres = 
                    mlir::cast<mlir::vllm_graph::ValueTensorType>(resType);
        if(Rankedres){
            llvm::ArrayRef<int64_t> shape = Rankedres.getSizes();
            std::vector<int64_t> shapeVec(shape.begin(), shape.end());
            mlir::Type elementType = Rankedres.getDtype();
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            elementType.print(os); // Prints the element type

            map["vllm_graph_type"] = "vllm_graph.vtensor";
            map["dtype"] = elementTypeName;
            map["output_shape"] = shapeVec;
            map["op_name"] = op->getName().getStringRef().str();

        }

        graph[ssa_id.str()] = map;
        for(mlir::Value operand : op->getOperands()){
            if(!opMap.count(operand)){
                llvm::errs() << "Operand : " << operand << " was not added.\n";
                throw std::runtime_error("op not added");
            }
            std::string operandSSA_id = opMap[operand];
            ValueType prevNodeData = graph[operandSSA_id]; 
            if(std::holds_alternative<std::unordered_map<std::string, NestedValueType>>(prevNodeData)){       
                std::unordered_map<std::string, NestedValueType> prev_map =
                                std::get<std::unordered_map<std::string, NestedValueType>>(prevNodeData);   
                if(!prev_map.count("next_nodes")){
                    std::vector<std::string> nextNode = {ssa_id.str()};
                    prev_map["next_nodes"] = nextNode;
                } else {
                    if(std::holds_alternative<std::vector<std::string>>(prev_map["next_nodes"])){
                        std::vector<std::string> nextNode = std::get<std::vector<std::string>>(prev_map["next_nodes"]);
                        nextNode.push_back(ssa_id.str());
                        prev_map["next_nodes"] = nextNode;
                    }
                }

                graph[operandSSA_id] = prev_map;
            }
                
        }
    }
}

GraphWriter::GraphWriter(){
    graph["entrypoint"] = std::vector<std::string>({});
}
void GraphWriter::build(mlir::OwningOpRef<mlir::ModuleOp> &module){
    module->walk([this](mlir::Operation *op) {
        // Print the operation name
        this->addOp(op);
    });
}