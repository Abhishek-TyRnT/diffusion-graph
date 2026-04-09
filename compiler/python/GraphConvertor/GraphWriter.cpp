
#include "GraphWriter.hpp"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "vllm_graph/Dialect/IR/vLLMGraphTypes.hpp"
#include "vllm_graph/Dialect/IR/vLLMGraphOps.hpp"

using namespace mlir;

template<> 
void GraphWriter::storeWeights<mlir::DenseElementsAttr>(mlir::DenseElementsAttr val, std::string ssa_id){
    Type type = val.getElementType();
    if(isa<IntegerType>(type))
    {   
        if(type.isInteger(64)){        
            std::vector<int64_t> denseVal(val.getValues<int64_t>().begin(), val.getValues<int64_t>().end());
            auto* int_proto = constData.add_integerweights();
            int_proto->set_name("weight_datasets" + ssa_id);
            
            // Add entire vector at once
            auto* values_field = int_proto->mutable_values();
            values_field->Reserve(denseVal.size());
            values_field->Add(denseVal.begin(), denseVal.end());
        } else {
            std::vector<int32_t> denseVal(val.getValues<int32_t>().begin(), val.getValues<int32_t>().end());
            auto* int_proto = constData.add_integerweights();
            int_proto->set_name("weight_datasets" + ssa_id);
            // Add entire vector at once
            auto* values_field = int_proto->mutable_values();
            values_field->Reserve(denseVal.size());
            values_field->Add(denseVal.begin(), denseVal.end());
        }

    } else { 
        std::vector<float> denseVal(val.getValues<float>().begin(), val.getValues<float>().end());
        auto* float_proto = constData.add_floatweights();
        float_proto->set_name("weight_datasets" + ssa_id);
        
        // Add entire vector at once
        auto* values_field = float_proto->mutable_values();
        values_field->Reserve(denseVal.size());
        values_field->Add(denseVal.begin(), denseVal.end());
    }
}

template<>
void GraphWriter::storeWeights<mlir::IntegerAttr>(mlir::IntegerAttr val, std::string ssa_id){
    int64_t value = val.getInt();
    
    auto *intconst_proto = constData.add_intconstants();
    intconst_proto->set_name("weight_datasets" + ssa_id);
    intconst_proto->set_values(value);

}

template<>
void GraphWriter::storeWeights<mlir::BoolAttr>(mlir::BoolAttr val, std::string ssa_id){
    bool value = val.getValue();

    auto *boolconst_proto = constData.add_boolconstants();
    boolconst_proto->set_name("weight_datasets" + ssa_id);
    boolconst_proto->set_values(value);
}

template<>
void GraphWriter::storeWeights<mlir::FloatAttr>(mlir::FloatAttr val, std::string ssa_id){
    llvm::APFloat AP_value = val.getValue();
    float value = AP_value.convertToFloat();
    auto *floatconst_proto = constData.add_floatconstants();
    floatconst_proto->set_name("weight_datasets" + ssa_id);
    floatconst_proto->set_values(value);
}

void GraphWriter::addOp(mlir::Operation *op, SubGraphMap &subGraph){

    if(mlir::isa<func::ReturnOp>(*op)){
        std::vector<std::string> results;
        for(auto operand : op->getOperands())
            results.push_back(opMap[operand]);

        subGraph["results"] = results;        
    }

    uint resCount = 0;
    for(mlir::Value res : op->getResults()){

        std::stringstream ssa_id;
        ssa_id << funcCount << "." << opCount << "." << resCount++;
        std::unordered_map<std::string, NestedValueType> map;
        mlir::Type resType = res.getType();
        //Case when it's a constant op 
        if(mlir::isa<mlir::arith::ConstantOp>(*op) || 
            mlir::isa<vllm_graph::ValueTensorLiteralOp>(*op)){
            TypedAttr attr;
            if(mlir::isa<mlir::arith::ConstantOp>(*op)){
                auto constOp = mlir::cast<mlir::arith::ConstantOp>(*op);
                attr = constOp.getValue();
            } else {
                auto constOp = mlir::cast<vllm_graph::ValueTensorLiteralOp>(*op);
                attr = constOp.getValue();
            }

            if(auto denseResourceAttr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(attr)){
                map["resource"] = denseResourceAttr.getRawHandle().getKey().str();
            } else if(auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
                storeWeights<mlir::DenseElementsAttr>(denseAttr, ssa_id.str());
            } else if(auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)){
                storeWeights<mlir::IntegerAttr>(intAttr, ssa_id.str());
            } else if(auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)){
                storeWeights<mlir::FloatAttr>(floatAttr, ssa_id.str());
            } else if(auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr)){
                storeWeights<mlir::BoolAttr>(boolAttr, ssa_id.str());
            }
            std::get<std::vector<std::string>>(subGraph["constants"]).push_back(ssa_id.str());
        } else if(mlir::isa<vllm_graph::ConstTupleOp>(*op)) {
            auto TupleOp = mlir::cast<vllm_graph::ConstTupleOp>(*op);
            auto attr = TupleOp.getValue();
            auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
            storeWeights<mlir::DenseElementsAttr>(denseAttr, ssa_id.str());
            std::get<std::vector<std::string>>(subGraph["constants"]).push_back(ssa_id.str());
        } else if(mlir::isa<vllm_graph::ConstantNoneOp>(op))
        {
            std::get<std::vector<std::string>>(subGraph["constants"]).push_back(ssa_id.str());
        } else if(mlir::isa<vllm_graph::ConstantStringOp>(op))
        {
            auto constOp = mlir::cast<vllm_graph::ConstantStringOp>(op);
            mlir::StringAttr attr = constOp.getValueAttr();
            map["value"] = attr.getValue().str();
            std::get<std::vector<std::string>>(subGraph["constants"]).push_back(ssa_id.str());
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
        else if(mlir::isa<vllm_graph::TupleType>(resType)){
            auto tupleType = mlir::cast<vllm_graph::TupleType>(resType);
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            tupleType.getContainedTypes()[0].print(os);
            map["vllm_graph_type"] = "tuple";
            map["dtype"] = elementTypeName;
            map["op_name"] = op->getName().getStringRef().str();
        }
        else if(mlir::isa<vllm_graph::ListType>(resType)){
            auto listType = mlir::cast<vllm_graph::ListType>(resType);
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            listType.getContainedType().print(os);
            map["vllm_graph_type"] = "list";
            map["dtype"] = elementTypeName;
            map["op_name"] = op->getName().getStringRef().str();
        }
        // Only for non tensor scalar type
        else{
            std::string elementTypeName;
            llvm::raw_string_ostream os(elementTypeName);
            resType.print(os);
            map["vllm_graph_type"] = "scalar";
            map["dtype"] = elementTypeName;
            map["op_name"] = op->getName().getStringRef().str();
        }
        opMap[res] = ssa_id.str();
        
        std::vector<std::string> input_nodes = {};
        for(mlir::Value operand : op->getOperands()){
            if(!opMap.count(operand)){
                llvm::errs() << "Operand : " << operand << " was not added.\n";
                throw std::runtime_error("op not added");
            }
            input_nodes.push_back(opMap[operand]);
            std::string operandSSA_id = opMap[operand];
            ValueType prevNodeData = subGraph[operandSSA_id]; 
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

                subGraph[operandSSA_id] = prev_map;
            }
                
        }
        map["input_nodes"] = input_nodes;
        subGraph[ssa_id.str()] = map;
    }

    opCount++;
}

GraphWriter::GraphWriter(std::string weightsPath) : weightsPath(weightsPath) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // file = H5::H5File(weightsPath, H5F_ACC_TRUNC);

}

void GraphWriter::build(mlir::OwningOpRef<mlir::ModuleOp> &module, 
                         llvm::DenseMap<StringRef, mlir::ArrayRef<char>> &dialectResourcesMap){

    // Print the module with resources
    mlir::OpPrintingFlags flags;
    
    SubGraphMap resourcePathMap;
    DenseResourceData.push_back(WeightShard(weightsPath, "weights_" + std::to_string(shard_index) + ".pb"));
   
    // Storing the float weights from the Dialect Resources Map into protobuf
    for (const auto& entry : dialectResourcesMap) {
        //TODO: Handle other dtypes of weights
        mlir::ArrayRef<char> blobData = entry.second;
        const float* floatData = reinterpret_cast<const float*>(blobData.data());
        size_t numFloats = blobData.size() / sizeof(float);
        std::vector<float> floatVector(floatData, floatData + numFloats);
        
        if(!DenseResourceData[shard_index].isSpaceAvailable(blobData.size())){
            shard_index++;
            DenseResourceData.push_back(WeightShard(weightsPath, "weights_" + std::to_string(shard_index) + ".pb"));
        }
       

        DenseResourceData[shard_index].updateSpaceOccupied(blobData.size());
        DenseWeights::WeightsData& weightsData = DenseResourceData[shard_index].getWeightsData();
        
        auto* float_proto = weightsData.add_floatweights();
        float_proto->set_name(entry.first.str());
        
        // Add entire vector at once
        auto* values_field = float_proto->mutable_values();
        values_field->Reserve(floatVector.size());
        values_field->Add(floatVector.begin(), floatVector.end());
       
        resourcePathMap[entry.first.str()] = DenseResourceData[shard_index].getShardName();
    }

    // SubGraphMap resourcesPathSubGraph;
    // for (const auto& entry : resourcePathMap)
    //     resourcesPathSubGraph[entry.first] = entry.second;
    graph["resources_path"] = resourcePathMap;
    // WeightShard DenseWeightShard(weightsPath + "/DenseAndScalarWeights.pb");
    graph["DenseAndScalarWeights"] = "DenseAndScalarWeights.pb";//DenseWeightShard.getShardPath();
    // constData = DenseWeightShard.getWeightsData();
    // DenseResourceData.push_back(DenseWeightShard);

    for(func::FuncOp funcOp : module->getOps<func::FuncOp>()){
        argCount = 0;
        opCount = 0;
        SubGraphMap subGraph;
        opMap.clear();
        subGraph["entrypoint"] = std::vector<std::string>({});
        subGraph["constants"] = std::vector<std::string>({});
        for(mlir::Value operand : funcOp.getArguments()){
            std::stringstream argName;
            argName << "arg" << argCount++;
            opMap[operand] = argName.str();
            std::get<std::vector<std::string>>(subGraph["entrypoint"]).push_back(argName.str());
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
                std::vector<std::string> input_nodes = {};
                map["input_nodes"] = input_nodes;
                map["op_name"] = "input_arg";
            }

            subGraph[argName.str()] = map;
        }

        Block &FuncBlock = funcOp.getBody().front();
        for(Operation &op : FuncBlock)
            addOp(&op, subGraph);

        StringRef funcName = funcOp.getName();
        graph[funcName.str()] = subGraph;
        funcCount++;
        // module->walk([this, &subGraph](mlir::Operation *op) {
        //     this->addOp(op, subGraph);
        // });
    }

}

void GraphWriter::closeFile(){
    
    for(WeightShard &weightShard : DenseResourceData){
        weightShard.saveShard();
    }
    
    std::ofstream output(weightsPath + "/DenseAndScalarWeights.pb", std::ios::binary);
    constData.SerializeToOstream(&output);
    output.close();
    // Clean up protobuf library
    google::protobuf::ShutdownProtobufLibrary();

}