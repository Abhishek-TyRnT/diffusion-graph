#ifndef VLLM_GRAPH_GRAPH_WRITER_H
#define VLLM_GRAPH_GRAPH_WRITER_H

#include <unordered_map>
#include <variant>
#include <any>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "weightBuffers.pb.h"

using NestedValueType = std::variant<std::string, 
                            std::vector<int64_t>, 
                            std::vector<std::string>, 
                            int, float>;
using ValueType = std::variant<std::string, 
                    std::unordered_map<std::string, NestedValueType>, std::vector<std::string>>;

using SubGraphMap = std::unordered_map<std::string, ValueType>;

class GraphWriter{
/*The Graph Writer class converts the vllm_graph IR to unordered map to subsequently 
    convert to json format.*/

private:
    
    std::unordered_map<std::string, SubGraphMap> graph;
    llvm::DenseMap<mlir::Value, std::string> opMap;
    
    DenseWeights::WeightsData constData;
    std::string weightsPath;
    //Counter to keep number of ops
    uint64_t opCount = 0;
    //Counter to keep number of args
    uint64_t argCount = 0; 
    //Counter to keep number of funcs
    uint64_t funcCount = 0;
    void addOp(mlir::Operation *op, SubGraphMap &subGraph);
    template<typename ElemType>
    void storeWeights(ElemType val, std::string ssa_id);

public:
    
    void build(mlir::OwningOpRef<mlir::ModuleOp> &module);
    GraphWriter(std::string weightsPath);
    void closeFile();
    std::unordered_map<std::string, SubGraphMap> getGraph(){ return graph; }
};
#endif


