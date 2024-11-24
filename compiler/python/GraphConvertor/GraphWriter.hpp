#ifndef VLLM_GRAPH_GRAPH_WRITER_H
#define VLLM_GRAPH_GRAPH_WRITER_H

#include <unordered_map>
#include <variant>
#include <any>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

class GraphWriter{
/*The Graph Writer class converts the vllm_graph IR to unordered map to subsequently 
    convert to json format.*/
using NestedValueType = std::variant<std::string, std::vector<int64_t>>;
using ValueType = std::variant<std::string, 
                    std::unordered_map<std::string, NestedValueType>>;

private:
    std::unordered_map<std::string, ValueType> graph;
    std::unordered_map<uint64_t, mlir::Operation*> opMap;
    //Counter to keep number of ops
    uint64_t opCount = 0;
    //Counter to keep number of args
    uint64_t argCount = 0; 
    void addOp(mlir::Operation *op);

public:
    
    void build(mlir::OwningOpRef<mlir::ModuleOp> &module);
    std::unordered_map<std::string, ValueType> getGraph(){ return graph; }
};
#endif


