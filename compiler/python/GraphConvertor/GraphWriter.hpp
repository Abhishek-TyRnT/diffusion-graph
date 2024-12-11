#ifndef VLLM_GRAPH_GRAPH_WRITER_H
#define VLLM_GRAPH_GRAPH_WRITER_H

#include "H5Cpp.h"  // HDF5 C++ API
#include <unordered_map>
#include <variant>
#include <any>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

using NestedValueType = std::variant<std::string, 
                            std::vector<int64_t>, 
                            std::vector<std::string>, 
                            int, float>;
using ValueType = std::variant<std::string, 
                    std::unordered_map<std::string, NestedValueType>, std::vector<std::string>>;
class GraphWriter{
/*The Graph Writer class converts the vllm_graph IR to unordered map to subsequently 
    convert to json format.*/

private:
    std::unordered_map<std::string, ValueType> graph;
    llvm::DenseMap<mlir::Value, std::string> opMap;
    H5::H5File file;
    //Counter to keep number of ops
    uint64_t opCount = 0;
    //Counter to keep number of args
    uint64_t argCount = 0; 
    void addOp(mlir::Operation *op);
    template<typename ElemType>
    void storeWeights(ElemType val, std::string ssa_id);

public:
    
    void build(mlir::OwningOpRef<mlir::ModuleOp> &module);
    GraphWriter(std::string weightsPath);
    void closeFile() { file.close(); }
    std::unordered_map<std::string, ValueType> getGraph(){ return graph; }
};
#endif


