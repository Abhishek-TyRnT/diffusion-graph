
#include "GraphWriter.hpp"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include <iostream>



template<> 
void GraphWriter::storeWeights<mlir::DenseElementsAttr>(mlir::DenseElementsAttr val, std::string ssa_id){
    std::vector<float> denseVal(val.getValues<float>().begin(), val.getValues<float>().end());
    hsize_t dims[1] = {denseVal.size()};
    H5::DataSpace dataspace(1, dims);
    // Create the dataset
    H5::DataSet dataset = file.createDataSet("weight_datasets" + ssa_id, H5::PredType::NATIVE_FLOAT, dataspace);
    dataset.write(denseVal.data(), H5::PredType::NATIVE_FLOAT);
    
}
void GraphWriter::addOp(mlir::Operation *op){

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
        //Case when it's a constant op 
        if(mlir::isa<mlir::arith::ConstantOp>(*op)){
            auto constOp = mlir::cast<mlir::arith::ConstantOp>(*op);
            auto attr = constOp.getValue();
            if (auto denseAttr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
                storeWeights<mlir::DenseElementsAttr>(denseAttr, ssa_id.str());
            }
            std::get<std::vector<std::string>>(graph["constants"]).push_back(ssa_id.str());
        }
        
        if(mlir::isa<mlir::vllm_graph::ValueTensorType>(resType)){
            auto Rankedres = mlir::cast<mlir::vllm_graph::ValueTensorType>(resType);
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

        else if(mlir::isa<mlir::RankedTensorType>(resType)){
            auto Rankedres = mlir::cast<mlir::RankedTensorType>(resType);
            llvm::ArrayRef<int64_t> shape = Rankedres.getShape();
            std::vector<int64_t> shapeVec(shape.begin(), shape.end());
            mlir::Type elementType = Rankedres.getElementType();
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            elementType.print(os); // Prints the element type

            map["vllm_graph_type"] = "tensor";
            map["dtype"] = elementTypeName;
            map["output_shape"] = shapeVec;
            map["op_name"] = op->getName().getStringRef().str();
        }
        // Only for non tensor scalar type
        else{
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            resType.print(os);
            map["dtype"] = elementTypeName;
            map["op_name"] = op->getName().getStringRef().str();
        }
        opMap[res] = ssa_id.str();
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

    opCount++;
}

GraphWriter::GraphWriter(){
    graph["entrypoint"] = std::vector<std::string>({});
    graph["constants"] = std::vector<std::string>({});

    file = H5::H5File("weights.h5", H5F_ACC_TRUNC);

}
void GraphWriter::build(mlir::OwningOpRef<mlir::ModuleOp> &module){
    module->walk([this](mlir::Operation *op) {
        // Print the operation name
        this->addOp(op);
    });
}